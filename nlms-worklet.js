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
    this.sumSqIn   = 0;
    this.sumSqOut  = 0;

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
      for (let j = 0; j < this.FL; j++) {
        const idx = (this.bi - j + this.FL) % this.FL;
        this.w[j] += step * e * this.buf[idx];
      }

      this.bi = (this.bi + 1) % this.FL;
      out[i]  = e;

      this.sumSqIn  += x * x;
      this.sumSqOut += e * e;
      if (++this.statCount >= this.statWin) {
        this.port.postMessage({
          rmsIn:  Math.sqrt(this.sumSqIn  / this.statWin),
          rmsOut: Math.sqrt(this.sumSqOut / this.statWin),
        });
        this.sumSqIn = this.sumSqOut = this.statCount = 0;
      }
    }
    return true;
  }
}
registerProcessor('nlms-processor', NLMSProcessor);
