package rlpark.plugin.rltoys.algorithms.control.sarsa;

import rlpark.plugin.rltoys.algorithms.functions.ParameterizedFunction;
import rlpark.plugin.rltoys.algorithms.functions.Predictor;
import rlpark.plugin.rltoys.algorithms.traces.ATraces;
import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;

@Monitor
public class Sarsa implements Predictor, ParameterizedFunction {
  private static final long serialVersionUID = 9030254074554565900L;
  @Monitor(level = 4)
  protected final Traces e;
  @Monitor(level = 4)
  protected final PVector q;
  protected double lambda;
  protected double gamma;
  protected double alpha;
  protected double delta;
  protected double v_t;
  protected double v_tp1;
  protected double minDeltaSeen=Double.MAX_VALUE;
  protected double maxDeltaSeen=Double.MIN_VALUE;

  public Sarsa(double alpha, double gamma, double lambda, int nbFeatures) {
    this(alpha, gamma, lambda, nbFeatures, new ATraces());
  }

  public Sarsa(double alpha, double gamma, double lambda, int nbFeatures, Traces prototype) {
    this(alpha, gamma, lambda, new PVector(nbFeatures), prototype);
  }

  public Sarsa(double alpha, double gamma, double lambda, PVector q, Traces prototype) {
    this.alpha = alpha;
    this.gamma = gamma;
    this.lambda = lambda;
    this.q = q;
    e = prototype.newTraces(q.getDimension());
  }

  public double update(RealVector phi_t, RealVector phi_tp1, double r_tp1) {
    if (phi_t == null)
      return initEpisode();
    v_tp1 = phi_tp1 != null ? q.dotProduct(phi_tp1) : 0;
    v_t = q.dotProduct(phi_t);
    delta = r_tp1 + gamma * v_tp1 - v_t;
    //minDeltaSeen=Math.min(delta, minDeltaSeen);
    //maxDeltaSeen=Math.max(delta, maxDeltaSeen);
    //System.out.println(delta+"\t"+minDeltaSeen+"\t"+maxDeltaSeen);
    e.update(gamma * lambda, phi_t);
    q.addToSelf(alpha * delta, e.vect());
    return delta;
  }

  protected double initEpisode() {
    e.clear();
    return 0.0;
  }

  @Override
  public double predict(RealVector phi_sa) {
    assert q.getDimension() == phi_sa.getDimension();
    return q.dotProduct(phi_sa);
  }

  @Override
  public PVector weights() {
    return q;
  }
}
