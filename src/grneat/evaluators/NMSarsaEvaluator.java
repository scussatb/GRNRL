package evaluators;

import java.util.Random;

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
import zephyr.plugin.core.api.synchronization.Clock;
import evolver.GRNGenome;
import grn.GRNModel;

public class NMSarsaEvaluator extends GRNGenomeEvaluator {

	public NMSarsaEvaluator() {
		numGRNInputs=1;
		numGRNOutputs=3;
		name="NMSarsa";
	}

	@Override
	public double evaluate(GRNGenome aGenome) {
		numEvaluations++;
		double fitness = evaluateGRN(buildGRNFromGenome(aGenome));
		aGenome.setNewFitness(fitness);
		return fitness;
	}
	
	public double evaluateGRN(GRNModel grn) {
		FunctionProjected2D valueFunctionDisplay;
		MountainCar problem;
		SarsaControl control;
		TileCodersNoHashing projector;
		Clock clock = new Clock("SarsaMountainCar");

		problem = new MountainCar(null);
		projector = new TileCodersNoHashing(problem.getObservationRanges());
		projector.addFullTilings(10, 10);
		projector.includeActiveFeature();
		TabularAction toStateAction = new TabularAction(problem.actions(), projector.vectorNorm(), projector.vectorSize());
		toStateAction.includeActiveFeature();
		double alpha = .15 / projector.vectorNorm();
		double gamma = 0.99;
		double lambda = .3;
		//System.out.print(numEvaluations+"\t: Evaluate: "+grn);
		GRNSarsa sarsa = new GRNSarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces(), grn);
		double epsilon = 0.01;
		Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
		Zephyr.advertise(clock, this);

		TRStep step = problem.initialize();
		int nbEpisode = 0;
		RealVector x_t = null;
		double fitness = 0;
		while (clock.tick() && nbEpisode<100) {
			BinaryVector x_tp1 = projector.project(step.o_tp1);
			Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
			x_t = Vectors.bufferedCopy(x_tp1, x_t);
			if (step.isEpisodeEnding() || step.time>5000) {
				//System.out.println(String.format("%d\t%d", nbEpisode, step.time));
				fitness += step.time;
				step = problem.initialize();
				x_t = null;
				nbEpisode++;
			} else {
				step = problem.step(action);
			}
		}
		//System.out.println(" = "+fitness/5);
		return -fitness/100;		
	}

	public static void main(String[] args) throws Exception {
		GRNModel grn = GRNModel.loadFromFile("NMSarsa/run_1420555820474228000/grn_18_-137.19.grn");
		
		double fitness = new NMSarsaEvaluator().evaluateGRN(grn);
		System.out.println(fitness);
	}

}
