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

    const delayMs = o.latencyMs || 20;
    this.delayN   = Math.max(1, Math.round(delayMs / 1000 * sampleRate));
    this.delayBuf = new Float32Array(this.delayN);
    this.delayIdx = 0;

    this.statCount = 0;
    this.statWin   = 4096;
    this.sumSqX    = 0;
    this.sumSqY    = 0;

    this.port.onmessage = (e) => {
      if (e.data.mu !== undefined) this.mu = e.data.mu;
    };
  }

  process(inputs, outputs) {
    const inp = inputs[0]?.[0];
    const out = outputs[0]?.[0];
    if (!inp || !out) return true;

    for (let i = 0; i < inp.length; i++) {
      const x = inp[i];

      // Signal retardé = approximation du bruit à l'oreille
      const d = this.delayBuf[this.delayIdx];
      this.delayBuf[this.delayIdx] = x;
      this.delayIdx = (this.delayIdx + 1) % this.delayN;

      // Mise à jour tampon circulaire
      this.buf[this.bi] = x;

      // Prédiction NLMS
      let y = 0, power = this.eps;
      for (let j = 0; j < this.FL; j++) {
        const idx = (this.bi - j + this.FL) % this.FL;
        y     += this.w[j] * this.buf[idx];
        power += this.buf[idx] * this.buf[idx];
      }

      // Erreur pour l'adaptation du filtre
      const e    = d - y;
      const step = this.mu / power;
      for (let j = 0; j < this.FL; j++) {
        const idx = (this.bi - j + this.FL) % this.FL;
        this.w[j] += step * e * this.buf[idx];
      }

      this.bi = (this.bi + 1) % this.FL;

      // ── CORRECTION CLÉ : on émet -y (prédiction inversée) ──────────
      // Le signal ANC à injecter dans les écouteurs est l'OPPOSÉ de ce
      // qu'on prédit arriver à l'oreille — pas le résidu e.
      out[i] = -y;

      // Stats : puissance entrée vs puissance signal ANC produit
      this.sumSqX += x * x;
      this.sumSqY += y * y;
      if (++this.statCount >= this.statWin) {
        this.port.postMessage({
          rmsIn:  Math.sqrt(this.sumSqX / this.statWin),
          rmsOut: Math.sqrt(this.sumSqY / this.statWin),
        });
        this.sumSqX = this.sumSqY = this.statCount = 0;
      }
    }
    return true;
  }
}
registerProcessor('nlms-processor', NLMSProcessor);
