package rlpark.plugin.rltoys.experiments.parametersweep.prediction.supervised;

import rlpark.plugin.rltoys.experiments.parametersweep.parameters.Parameters;
import rlpark.plugin.rltoys.experiments.parametersweep.prediction.PredictionLearnerFactory;
import rlpark.plugin.rltoys.experiments.parametersweep.prediction.PredictionParameters;
import rlpark.plugin.rltoys.experiments.parametersweep.prediction.PredictionProblemFactory;
import rlpark.plugin.rltoys.experiments.parametersweep.prediction.PredictionSweepContext;
import rlpark.plugin.rltoys.experiments.parametersweep.prediction.PredictorEvaluator;

public class SupervisedSweepContext extends PredictionSweepContext {
  private static final long serialVersionUID = -3512756368836614504L;

  public SupervisedSweepContext(PredictionProblemFactory problemFactory, PredictionLearnerFactory learnerFactory) {
    super(problemFactory, learnerFactory);
  }

  @Override
  public PredictorEvaluator createPredictorEvaluator(Parameters parameters) {
    int nbEvaluationSteps = PredictionParameters.nbEvaluationSteps(parameters);
    int nbPerformanceCheckpoints = PredictionParameters.nbPerformanceCheckpoint(parameters);
    return new SupervisedErrorMonitor(nbPerformanceCheckpoints, nbEvaluationSteps);
  }
}
