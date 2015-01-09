package rlpark.example.demos.learning;

import java.util.Random;

import rlpark.plugin.rltoys.agents.functions.FunctionProjected2D;
import rlpark.plugin.rltoys.agents.functions.ValueFunction2D;
import rlpark.plugin.rltoys.algorithms.control.acting.EpsilonGreedy;
import rlpark.plugin.rltoys.algorithms.control.sarsa.Sarsa;
import rlpark.plugin.rltoys.algorithms.control.sarsa.SarsaControl;
import rlpark.plugin.rltoys.algorithms.functions.stateactions.TabularAction;
import rlpark.plugin.rltoys.algorithms.representations.tilescoding.TileCodersNoHashing;
import rlpark.plugin.rltoys.algorithms.traces.RTraces;
import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.envio.rl.TRStep;
import rlpark.plugin.rltoys.math.vector.BinaryVector;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.Vectors;
import rlpark.plugin.rltoys.problems.mountaincar.MountainCar;
import zephyr.plugin.core.api.Zephyr;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;
import zephyr.plugin.core.api.synchronization.Clock;

@Monitor
public class SarsaMountainCar  {
  final FunctionProjected2D valueFunctionDisplay;
  public final MountainCar problem;
  public final SarsaControl control;
  public final TileCodersNoHashing projector;
  public final Clock clock = new Clock("SarsaMountainCar");

  public SarsaMountainCar() {
    problem = new MountainCar(null);
    projector = new TileCodersNoHashing(problem.getObservationRanges());
    projector.addFullTilings(10, 10);
    projector.includeActiveFeature();
    TabularAction toStateAction = new TabularAction(problem.actions(), projector.vectorNorm(), projector.vectorSize());
    toStateAction.includeActiveFeature();
    double alpha = .15 / projector.vectorNorm();
    double gamma = 0.99;
    double lambda = .3;
    Sarsa sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces());
    double epsilon = 0.01;
    Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
    control = new SarsaControl(acting, toStateAction, sarsa);
    valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
    Zephyr.advertise(clock, this);
  }

  public double run() {
    TRStep step = problem.initialize();
    int nbEpisode = 0;
    double fit=0;
    RealVector x_t = null;
    while (clock.tick() && nbEpisode<100) {
      BinaryVector x_tp1 = projector.project(step.o_tp1);
      Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
      x_t = Vectors.bufferedCopy(x_tp1, x_t);
      if (step.isEpisodeEnding()) {
          //System.out.println(String.format("Episode %d: %d steps", nbEpisode, step.time));
          System.out.println(String.format("%d\t%d", nbEpisode, step.time));
    	fit+=step.time;
        step = problem.initialize();
        x_t = null;
        nbEpisode++;
      } else
        step = problem.step(action);
    }
    return fit/100;
  }

  public static void main(String[] args) {
	  double fit=new SarsaMountainCar().run();
	  System.out.println(fit);
	    
  }
}
