class NLMSProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.FL  = o.filterLength || 256;
    this.mu  = o.mu || 0.3;
    this.eps = 1e-6;
    this.w   = new Float32Array(this.FL);
    this.buf = new Float32Array(this.FL);
    this.bi  = 0;
    this.outputScale = o.outputScale || 1.0;

    // Tampon de délai ajustable (clé pour l'alignement de phase)
    this.setDelay(o.delayMs || 8);

    // Détecteur de stationnarité (speech gate)
    // Compare RMS court terme (10ms) vs long terme (200ms)
    this.shortWin = Math.round(0.010 * sampleRate);
    this.longWin  = Math.round(0.200 * sampleRate);
    this.shortSum = 0;
    this.longSum  = 0;
    this.shortBuf = new Float32Array(this.shortWin);
    this.longBuf  = new Float32Array(this.longWin);
    this.shortIdx = 0;
    this.longIdx  = 0;
    // Gate : sortie lissée (évite les clics)
    this.gateVal  = 1.0;
    this.gateAtk  = 0.001;  // attaque rapide (mute en ~1ms)
    this.gateRel  = 0.050;  // relâchement lent (retour en ~50ms)

    // Stats
    this.statCount = 0;
    this.statWin   = 4096;
    this.sumSqX    = 0;
    this.sumSqY    = 0;

    this.port.onmessage = (e) => {
      if (e.data.mu          !== undefined) this.mu = e.data.mu;
      if (e.data.outputScale !== undefined) this.outputScale = e.data.outputScale;
      if (e.data.delayMs     !== undefined) this.setDelay(e.data.delayMs);
      if (e.data.speechGate  !== undefined) this.speechGateEnabled = e.data.speechGate;
    };
    this.speechGateEnabled = true;
  }

  setDelay(ms) {
    this.delayN   = Math.max(1, Math.round(ms / 1000 * sampleRate));
    this.delayBuf = new Float32Array(this.delayN);
    this.delayIdx = 0;
  }

  process(inputs, outputs) {
    const inp = inputs[0]?.[0];
    const out = outputs[0]?.[0];
    if (!inp || !out) return true;

    for (let i = 0; i < inp.length; i++) {
      const x = inp[i];

      // ── Détection stationnarité ─────────────────────────────────
      const xsq = x * x;
      this.shortSum -= this.shortBuf[this.shortIdx];
      this.shortBuf[this.shortIdx] = xsq;
      this.shortSum += xsq;
      this.shortIdx = (this.shortIdx + 1) % this.shortWin;

      this.longSum -= this.longBuf[this.longIdx];
      this.longBuf[this.longIdx] = xsq;
      this.longSum += xsq;
      this.longIdx = (this.longIdx + 1) % this.longWin;

      const rmsShort = Math.sqrt(this.shortSum / this.shortWin + 1e-12);
      const rmsLong  = Math.sqrt(this.longSum  / this.longWin  + 1e-12);

      // Si court terme >> long terme : signal non-stationnaire (voix, claquement)
      const isStationary = rmsShort < rmsLong * 2.0;
      const targetGate = (this.speechGateEnabled && !isStationary) ? 0.0 : 1.0;

      // Lissage du gate (évite les clics)
      if (targetGate < this.gateVal)
        this.gateVal = this.gateVal * (1 - this.gateAtk) + targetGate * this.gateAtk;
      else
        this.gateVal = this.gateVal * (1 - this.gateRel) + targetGate * this.gateRel;

      // ── NLMS ───────────────────────────────────────────────────
      // Signal retardé (cible de prédiction)
      const d = this.delayBuf[this.delayIdx];
      this.delayBuf[this.delayIdx] = x;
      this.delayIdx = (this.delayIdx + 1) % this.delayN;

      this.buf[this.bi] = x;

      let y = 0, power = this.eps;
      for (let j = 0; j < this.FL; j++) {
        const idx = (this.bi - j + this.FL) % this.FL;
        y     += this.w[j] * this.buf[idx];
        power += this.buf[idx] * this.buf[idx];
      }

      const e    = d - y;
      const step = this.mu / power;

      // N'adapte les coefficients que si signal stationnaire
      if (this.gateVal > 0.5) {
        for (let j = 0; j < this.FL; j++) {
          const idx = (this.bi - j + this.FL) % this.FL;
          this.w[j] += step * e * this.buf[idx];
        }
      }

      this.bi = (this.bi + 1) % this.FL;

      // ── Sortie ANC : prédiction inversée + gate + scale ────────
      out[i] = -y * this.outputScale * this.gateVal;

      this.sumSqX += x * x;
      this.sumSqY += y * y;
      if (++this.statCount >= this.statWin) {
        this.port.postMessage({
          rmsIn:  Math.sqrt(this.sumSqX / this.statWin),
          rmsOut: Math.sqrt(this.sumSqY / this.statWin),
          gate:   this.gateVal,
        });
        this.sumSqX = this.sumSqY = this.statCount = 0;
      }
    }
    return true;
  }
}
registerProcessor('nlms-processor', NLMSProcessor);
