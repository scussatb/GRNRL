package rlpark.example.demos.learning;

import grn.GRNModel;
import grn.GRNProtein;

import java.util.Random;
import java.util.Vector;

import rlpark.plugin.rltoys.agents.functions.FunctionProjected2D;
import rlpark.plugin.rltoys.agents.functions.ValueFunction2D;
import rlpark.plugin.rltoys.algorithms.control.acting.EpsilonGreedy;
import rlpark.plugin.rltoys.algorithms.control.sarsa.GRNSarsa;
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
public class GRNSarsaMountainCar implements Runnable {
	final FunctionProjected2D valueFunctionDisplay;
	private final MountainCar problem;
	private final SarsaControl control;
	private final TileCodersNoHashing projector;
	private final Clock clock = new Clock("SarsaMountainCar");

	public GRNSarsaMountainCar(GRNModel grn) {
		problem = new MountainCar(null);
		projector = new TileCodersNoHashing(problem.getObservationRanges());
		projector.addFullTilings(10, 10);
		projector.includeActiveFeature();
		TabularAction toStateAction = new TabularAction(problem.actions(), projector.vectorNorm(), projector.vectorSize());
		toStateAction.includeActiveFeature();
		double alpha = .15 / projector.vectorNorm();
		double gamma = 0.99;
		double lambda = .3;
		GRNSarsa sarsa = new GRNSarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces(), grn);
		double epsilon = 0.01;
		Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
		Zephyr.advertise(clock, this);
	}

	@Override
	public void run() {
		TRStep step = problem.initialize();
		int nbEpisode = 0;
		RealVector x_t = null;
		while (clock.tick()) {
			BinaryVector x_tp1 = projector.project(step.o_tp1);
			Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
			x_t = Vectors.bufferedCopy(x_tp1, x_t);
			if (step.isEpisodeEnding()) {
				System.out.println(String.format("Episode %d: %d steps", nbEpisode, step.time));
				step = problem.initialize();
				x_t = null;
				nbEpisode++;
			} else
				step = problem.step(action);
		}
	}

	public static void main(String[] args) {
		Vector<GRNProtein> proteins = new Vector<GRNProtein>();
		proteins.add(new GRNProtein((int)(Math.random()*GRNProtein.IDSIZE), GRNProtein.OUTPUT_PROTEIN, 0.0, (int)(Math.random()*GRNProtein.IDSIZE), (int)(Math.random()*GRNProtein.IDSIZE)));
		proteins.add(new GRNProtein((int)(Math.random()*GRNProtein.IDSIZE), GRNProtein.OUTPUT_PROTEIN, 0.0, (int)(Math.random()*GRNProtein.IDSIZE), (int)(Math.random()*GRNProtein.IDSIZE)));
		proteins.add(new GRNProtein((int)(Math.random()*GRNProtein.IDSIZE), GRNProtein.OUTPUT_PROTEIN, 0.0, (int)(Math.random()*GRNProtein.IDSIZE), (int)(Math.random()*GRNProtein.IDSIZE)));
		int nRegul=(int)(Math.random()*5)+1;
		for (int i=0; i<nRegul; i++) {
			proteins.add(new GRNProtein((int)(Math.random()*GRNProtein.IDSIZE), GRNProtein.REGULATORY_PROTEIN, 0.0, (int)(Math.random()*GRNProtein.IDSIZE), (int)(Math.random()*GRNProtein.IDSIZE)));
		}
		new GRNSarsaMountainCar(new GRNModel(proteins, 1.0, 1.0)).run();
	}
}
