package evaluators;

import java.util.Random;
import java.util.Vector;

import javax.print.DocFlavor.STRING;

import rlpark.plugin.rltoys.agents.functions.FunctionProjected2D;
import rlpark.plugin.rltoys.agents.functions.ValueFunction2D;
import rlpark.plugin.rltoys.agents.rl.LearnerAgentFA;
import rlpark.plugin.rltoys.algorithms.control.ControlLearner;
import rlpark.plugin.rltoys.algorithms.control.acting.EpsilonGreedy;
import rlpark.plugin.rltoys.algorithms.control.actorcritic.onpolicy.Actor;
import rlpark.plugin.rltoys.algorithms.control.actorcritic.onpolicy.AverageRewardActorCritic;
import rlpark.plugin.rltoys.algorithms.control.qlearning.QLearning;
import rlpark.plugin.rltoys.algorithms.control.qlearning.QLearningControl;
import rlpark.plugin.rltoys.algorithms.control.sarsa.GRNSarsa;
import rlpark.plugin.rltoys.algorithms.control.sarsa.Sarsa;
import rlpark.plugin.rltoys.algorithms.control.sarsa.SarsaControl;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.PolicyDistribution;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.helpers.ScaledPolicyDistribution;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.structures.NormalDistributionScaled;
import rlpark.plugin.rltoys.algorithms.functions.stateactions.TabularAction;
import rlpark.plugin.rltoys.algorithms.functions.states.Projector;
import rlpark.plugin.rltoys.algorithms.predictions.td.OnPolicyTD;
import rlpark.plugin.rltoys.algorithms.predictions.td.TDLambda;
import rlpark.plugin.rltoys.algorithms.representations.discretizer.partitions.AbstractPartitionFactory;
import rlpark.plugin.rltoys.algorithms.representations.tilescoding.TileCodersNoHashing;
import rlpark.plugin.rltoys.algorithms.traces.RTraces;
import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.envio.rl.TRStep;
import rlpark.plugin.rltoys.experiments.runners.Runner;
import rlpark.plugin.rltoys.experiments.runners.AbstractRunner.RunnerEvent;
import rlpark.plugin.rltoys.junit.problems.puddleworld.PuddleWorldTest;
import rlpark.plugin.rltoys.math.ranges.Range;
import rlpark.plugin.rltoys.math.vector.BinaryVector;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import rlpark.plugin.rltoys.math.vector.implementations.Vectors;
import rlpark.plugin.rltoys.problems.helicopter.Helicopter;
import rlpark.plugin.rltoys.problems.mazes.Maze;
import rlpark.plugin.rltoys.problems.mazes.MazeValueFunction;
import rlpark.plugin.rltoys.problems.mazes.Mazes;
import rlpark.plugin.rltoys.problems.mountaincar.MountainCar;
import rlpark.plugin.rltoys.problems.pendulum.SwingPendulum;
import rlpark.plugin.rltoys.problems.puddleworld.LocalFeatureSumFunction;
import rlpark.plugin.rltoys.problems.puddleworld.PuddleWorld;
import rlpark.plugin.rltoys.problems.puddleworld.RewardWhenTerminated;
import rlpark.plugin.rltoys.problems.puddleworld.SmoothPuddle;
import rlpark.plugin.rltoys.problems.puddleworld.TargetReachedL1NormTermination;
import rlpark.plugin.rltoys.problems.puddleworld.TargetReachedL2NormTermination;
import rlpark.plugin.rltoys.problems.puddleworld.TerminationFunction;
import rlpark.plugin.rltoysview.tests.internal.puddleworld.TestPuddleWorldRunnable;
import zephyr.plugin.core.api.Zephyr;
import zephyr.plugin.core.api.signals.Listener;
import zephyr.plugin.core.api.synchronization.Clock;
import evolver.GRNGenome;
import grn.GRNModel;

public class NMSarsaEvaluator extends GRNGenomeEvaluator {
	public double fitness;
	public int nEpisode;
	public boolean displayEpisodes = false;
	public Vector<String> problems = new Vector<String>();

	// e.rng = new java.util.Random( e.randomSeed );
	public Random rngRL;

	double alpha = 0.1;
	double gamma = 0.9;
	double lambda = 0.1;
	double epsilon = 0.1;
	int randomSeed = 0;

	public NMSarsaEvaluator() {
		numGRNInputs = 3;
		numGRNOutputs = 4;
		nonCacheable=true;
		// problems.add("MountainCar");
		// problems.add( problem );
		// problems = inProblems;
		// name="NMSarsa"+problems.toString();
		name = "NMSarsa_uninitialized";
		alpha = 0.15;
		gamma = 1.0;
		lambda = 0.6;
		epsilon = 0.3;
	}

	public void setSarsaDefaults(String problem) {
		if (problems.contains("MountainCar")) {
			alpha = .15;// / projector.vectorNorm();
			gamma = 0.99;
			lambda = .3;
			epsilon = 0.01;
		}
		if (problems.contains("Maze")) {
			alpha = 0.15;
			gamma = 1.0;
			lambda = 0.6;
			epsilon = 0.3;
		}
		if (problems.contains("ActorCriticPendulum")) {
			alpha = 0.15;
			gamma = 1.0;
			lambda = 0.6;
			epsilon = 0.3;
		}
		if (problems.contains("PuddleWorld")) {
			alpha = .15;
			gamma = 1.0;
			lambda = 0.6;
			epsilon = 0.3;
		}

	}

	public void readArgs(String[] args) {
		for (int k = 0; k < args.length; k++) {
			// System.err.print("\t" + args[k]);

			if (args[k].compareTo("alpha") == 0) {
				alpha = Double.parseDouble(args[k + 1]);
			} else if (args[k].compareTo("gamma") == 0) {
				gamma = Double.parseDouble(args[k + 1]);
			} else if (args[k].compareTo("lambda") == 0) {
				lambda = Double.parseDouble(args[k + 1]);
			} else if (args[k].compareTo("epsilon") == 0) {
				epsilon = Double.parseDouble(args[k + 1]);
			} else if (args[k].compareTo("problems") == 0) {
				while (args[k + 1].compareTo("endProblems") != 0) {
					problems.add(args[k + 1]);
					k++;
				}
				name = "NMSarsa" + problems.toString();
			} else if (args[k].compareTo("randomSeed") == 0) {
				randomSeed = Integer.parseInt(args[k + 1]);
				rngRL = new java.util.Random(randomSeed);
			}

		}
		System.out.println("alpha = " + alpha);
		System.out.println("gamma = " + gamma);
		System.out.println("lambda = " + lambda);
		System.out.println("epsilon = " + epsilon);
		System.out.println("randomSeed = " + randomSeed);
	}

	@Override
	public double evaluate(GRNGenome aGenome) {
		numEvaluations++;
		double fitness = evaluateGRN(buildGRNFromGenome(aGenome), problems);
		aGenome.setNewFitness(fitness);
		return fitness;
	}

	public double evaluateGRN(GRNModel grn, Vector<String> problems) {
		double fitness = 0;
		if (problems.contains("MountainCar")) {
			fitness += evaluateMountainCar(grn);
		}
		if (problems.contains("Maze")) {
			fitness += evaluateMaze(grn);
		}
		if (problems.contains("ActorCriticPendulum")) {
			fitness += evaluateActorCriticPendulum(grn);
		}
		if (problems.contains("PuddleWorld")) {
			fitness += evaluatePuddleWorld(grn);
		}
		// System.out.println(grn+" : "+fitness);
		return fitness;
	}

	public double evaluateMountainCar(GRNModel grn) {

		FunctionProjected2D valueFunctionDisplay;
		MountainCar problem;
		SarsaControl control;
		TileCodersNoHashing projector;
		Clock clock = new Clock("SarsaMountainCar");

		problem = new MountainCar(null);
		projector = new TileCodersNoHashing(problem.getObservationRanges());
		projector.addFullTilings(10, 10);
		projector.includeActiveFeature();

		TabularAction toStateAction = new TabularAction(problem.actions(),
				projector.vectorNorm(), projector.vectorSize());
		toStateAction.includeActiveFeature();
//		alpha /= projector.vectorNorm();
		// double alpha = .15 / projector.vectorNorm();
		// double gamma = 0.99;
		// double lambda = .3;
		// double alpha = 0.021468632628537543;
		// double gamma = 0.9785215963986705;
		// double lambda = 0.0;
		// System.out.print(numEvaluations+"\t: Evaluate: "+grn);
		Sarsa sarsa;
		if (grn != null) {
			setSarsaDefaults("MountainCar");
			sarsa = new GRNSarsa(alpha, gamma, lambda,
					toStateAction.vectorSize(), new RTraces(), grn,
					projector.vectorNorm());
			((GRNSarsa) sarsa).minDelta = -33.5031963543928;
			((GRNSarsa) sarsa).maxDelta = 17.841888320550;
		} else {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(),
					new RTraces());
		}
		// double epsilon = 0.01;
		// Policy acting = new EpsilonGreedy(new Random(0), problem.actions(),
		// toStateAction, sarsa, epsilon);
		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
		Zephyr.advertise(clock, this);

		TRStep step = problem.initialize();
		int nbEpisode = 0;
		RealVector x_t = null;
		fitness = 0;
		while (clock.tick() && nbEpisode < 100) {
			BinaryVector x_tp1 = projector.project(step.o_tp1);
			Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
			x_t = Vectors.bufferedCopy(x_tp1, x_t);
			if (step.isEpisodeEnding() || step.time > 5000) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%d", nbEpisode,
							step.time));
				fitness += step.time;
				step = problem.initialize();
				x_t = null;
				nbEpisode++;
			} else {
				step = problem.step(action);
			}
		}
		// System.out.println(" = "+fitness/5);
		return -fitness / 100;

	}

	public double evaluateMaze(GRNModel grn) {
		MazeValueFunction mazeValueFunction;
		Maze problem = Mazes.createBookMaze();
		ControlLearner control;
		Clock clock = new Clock("QLearningMaze");
		Projector projector;
		PVector occupancy;
		LearnerAgentFA agent;

		projector = problem.getMarkovProjector();
		occupancy = new PVector(projector.vectorSize());
		TabularAction toStateAction = new TabularAction(problem.actions(),
				projector.vectorNorm(), projector.vectorSize());
		alpha /= projector.vectorNorm();
		// double alpha = .15 / projector.vectorNorm();
		// double gamma = 1.0;
		// double lambda = 0.6;
		// double alpha = 0.021468632628537543;
		// double gamma = 0.9785215963986705;
		// double lambda = 0.0;

		Sarsa sarsa;
		if (grn == null) {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(),
					new RTraces());
		} else {
			setSarsaDefaults("Maze");
			sarsa = new GRNSarsa(alpha, gamma, lambda,
					toStateAction.vectorSize(), new RTraces(), grn,
					projector.vectorNorm());
			((GRNSarsa) sarsa).minDelta = -5.285246761385281;
			((GRNSarsa) sarsa).maxDelta = 3.4911249865196456;

		}
		// double epsilon = 0.3;
		// Policy acting = new EpsilonGreedy(new Random(0), problem.actions(),
		// toStateAction, sarsa, epsilon);
		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, projector);
		mazeValueFunction = new MazeValueFunction(problem, sarsa,
				toStateAction, acting);
		Zephyr.advertise(clock, this);

		// double fitness = 0;
		nEpisode = 0;
		Runner runner = new Runner(problem, agent, 100, 10000);
		fitness = 0;
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%d",
							eventInfo.nbEpisodeDone, eventInfo.step.time));
				nEpisode++;
				fitness += eventInfo.step.time;
			}
		});
		while (clock.tick() && nEpisode < 100) {
			runner.step();
			occupancy.addToSelf(agent.lastState());
		}

		// System.out.println(" = "+fitness/5);
		return -fitness / 100;
	}

	public double evaluatePuddleWorld(GRNModel grn) {
		FunctionProjected2D valueFunction;
		double reward;
		Clock clock = new Clock("PuddleWorld");
		LearnerAgentFA agent;
		Runner runner;

		// Random random = new Random(0);
		Random random = rngRL;
		Range observationRange = new Range(-50, 50);
		Range actionRange = new Range(-1, 1);
		double noise = .1;
		PuddleWorld world = new PuddleWorld(random, 2, observationRange,
				actionRange, noise);
		world.setStart(new double[] { -49, -49 });
		TargetReachedL2NormTermination termFunc = new TargetReachedL2NormTermination(
				new double[] { 49, 49 }, actionRange.max() + 2 * noise);
		world.setTermination(termFunc);
		double weights[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 15 };
		RewardWhenTerminated features[] = {
				new RewardWhenTerminated(new TargetReachedL2NormTermination(
						new double[] { -45, -45 }, actionRange.max() + 2
						* noise)),
						new RewardWhenTerminated(new TargetReachedL2NormTermination(
								new double[] { -40, -45 }, actionRange.max() + 2
								* noise)),
								new RewardWhenTerminated(
										new TargetReachedL2NormTermination(new double[] { -35,
												30 }, actionRange.max() + 2 * noise)),
												new RewardWhenTerminated(
														new TargetReachedL2NormTermination(new double[] { -15,
																10 }, actionRange.max() + 2 * noise)),
																new RewardWhenTerminated(new TargetReachedL2NormTermination(
																		new double[] { 0, 0 }, actionRange.max() + 2 * noise)),
																		new RewardWhenTerminated(new TargetReachedL2NormTermination(
																				new double[] { 5, 15 }, actionRange.max() + 2 * noise)),
																				new RewardWhenTerminated(new TargetReachedL2NormTermination(
																						new double[] { 25, 30 }, actionRange.max() + 2 * noise)),
																						new RewardWhenTerminated(new TargetReachedL2NormTermination(
																								new double[] { 40, 35 }, actionRange.max() + 2 * noise)),
																								new RewardWhenTerminated(new TargetReachedL2NormTermination(
																										new double[] { 45, 45 }, actionRange.max() + 2 * noise)),

																										new RewardWhenTerminated(new TargetReachedL2NormTermination(
																												new double[] { 49, 49 }, actionRange.max() + 2 * noise)) };
		world.setRewardFunction(new LocalFeatureSumFunction(weights, features,
				-0.1));
		TileCodersNoHashing tileCoders = new TileCodersNoHashing(
				world.getObservationRanges());
		((AbstractPartitionFactory) tileCoders.discretizerFactory()).setRandom(
				random, .2);
		tileCoders.addFullTilings(10, 10);
		// double alpha = .15 / tileCoders.vectorNorm();
		// double gamma = 1.0;
		// double lambda = 0.6;
		//		alpha /= tileCoders.vectorNorm();
		double vectorNorm = tileCoders.vectorNorm();
		int vectorSize = tileCoders.vectorSize();
		TabularAction toStateAction = new TabularAction(world.actions(),
				tileCoders.vectorNorm(), tileCoders.vectorSize());
		Sarsa sarsa;
		if (grn == null) {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(),
					new RTraces());
		} else {
			setSarsaDefaults("PuddleWorld");
			sarsa = new GRNSarsa(alpha, gamma, lambda,
					toStateAction.vectorSize(), new RTraces(), grn,
					tileCoders.vectorNorm());
			((GRNSarsa) sarsa).minDelta = -6442.566526517287;
			((GRNSarsa) sarsa).maxDelta = 10611.01805647782;
		}
		// double epsilon = 0.3;
		// Policy acting = new EpsilonGreedy(new Random(0), world.actions(),
		// toStateAction, sarsa, epsilon);
		Policy acting = new EpsilonGreedy(rngRL, world.actions(),
				toStateAction, sarsa, epsilon);
		ControlLearner control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, tileCoders);
		valueFunction = new ValueFunction2D(tileCoders, world, sarsa);
		runner = new Runner(world, agent, 100, 1000);
		nEpisode = 0;
		fitness = 0;
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%f",
							eventInfo.nbEpisodeDone, eventInfo.episodeReward));
				nEpisode++;
				fitness += eventInfo.episodeReward;
			}
		});

		Zephyr.advertise(clock, this);

		while (clock.tick() && nEpisode < 100) {
			runner.step();
		}

		return fitness / 100;

	}

	public double evaluateActorCriticPendulum(GRNModel grn) {
		FunctionProjected2D valueFunction;
		double reward;
		SwingPendulum problem;
		Clock clock = new Clock("ActorCriticPendulum");
		LearnerAgentFA agent;
		Runner runner;

		// Random random = new Random(0);
		Random random = rngRL;
		problem = new SwingPendulum(null, false);
		TileCodersNoHashing tileCoders = new TileCodersNoHashing(
				problem.getObservationRanges());
		((AbstractPartitionFactory) tileCoders.discretizerFactory()).setRandom(
				random, .2);
		tileCoders.addFullTilings(10, 10);
//		alpha /= tileCoders.vectorNorm();
		// double alpha = .15 / tileCoders.vectorNorm();
		// double gamma = 1.0;
		// double lambda = 0.6;
		double vectorNorm = tileCoders.vectorNorm();
		int vectorSize = tileCoders.vectorSize();
		TabularAction toStateAction = new TabularAction(problem.actions(),
				tileCoders.vectorNorm(), tileCoders.vectorSize());
		Sarsa sarsa;
		if (grn == null) {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(),
					new RTraces());
		} else {
			setSarsaDefaults("ActorCriticPendulum");
			sarsa = new GRNSarsa(alpha, gamma, lambda,
					toStateAction.vectorSize(), new RTraces(), grn,
					tileCoders.vectorNorm());
			((GRNSarsa) sarsa).minDelta = -318.8166143617434;
			((GRNSarsa) sarsa).maxDelta = 321.8407343516526;
		}
		// double epsilon = 0.3;
		// Policy acting = new EpsilonGreedy(new Random(0), problem.actions(),
		// toStateAction, sarsa, epsilon);
		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		ControlLearner control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, tileCoders);
		valueFunction = new ValueFunction2D(tileCoders, problem, sarsa);
		runner = new Runner(problem, agent, 100, 5000);
		nEpisode = 0;
		fitness = 0;
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%f",
							eventInfo.nbEpisodeDone, eventInfo.episodeReward));
				nEpisode++;
				fitness += eventInfo.episodeReward;
			}
		});

		Zephyr.advertise(clock, this);

		while (clock.tick() && nEpisode < 100) {
			runner.step();
		}

		return fitness / 100;

	}

	public static void main(String[] args) throws Exception {
//		GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/NMSarsa_PuddleWorld/run_1421850433035169000/grn_7_1579.530000000022.grn");
//		GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-13/NMSarsa_Maze-ActorCriticPendulum/run_769851943476400/grn_81_2504.806138294125.grn");
		GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/NMSarsa_Maze/run_1421926719262292000/grn_67_-33.27.grn");		
		float sumGRN=0;
		float sumSarsa=0;
		int maxEval=100;
		for (int i=0; i<maxEval; i++) {
			NMSarsaEvaluator eval = new NMSarsaEvaluator();
			// eval.readArgs( args );
			eval.displayEpisodes = false;
			// eval.setSarsaDefaults(eval.problems.get(0));
			
//			eval.problems.add("PuddleWorld");

//			eval.problems.add("MountainCar");
//			eval.alpha =  0.07142857142854546;
//			eval.gamma = 1.0;
//			eval.lambda = 0.928571428571;
//			eval.epsilon = 0.01;
			eval.problems.add("Maze");
//			eval.alpha =  1.0;
//			eval.gamma = 0.928571428571;
//			eval.lambda = 0.928571428571;
//			eval.epsilon = 0.01;
//			eval.problems.add("PuddleWorld");
			eval.alpha =   0.0571428571429;
			eval.gamma = 0.928571428571;
			eval.lambda = 0.5;
			eval.epsilon = 0.01;
//			eval.problems.add("ActorCriticPendulum");
//			eval.alpha =   0.05;
//			eval.gamma = 0.928571428571;
//			eval.lambda = 0.785714285714;
//			eval.epsilon = 0.01;
		
			
			eval.rngRL = new java.util.Random(i*1000);

			// System.out.println("====   Evaluating SARSA  ====");
//			double fSarsa = eval.evaluateGRN(null, eval.problems);
			//			System.out.println("fSarsa = " + fSarsa);
			// System.out.println("\n==== Evaluating GRNSARSA ====");
			double fGRN = eval.evaluateGRN(grn, eval.problems);
			sumGRN+=fGRN;
//			sumSarsa+=fSarsa;
			System.out.println(i+"\t"+fGRN/*+"\t"+fSarsa/**/);
			// System.out.println("\n====       Averages      ====\nGRNSarsa\tSarsa\n"+fGRN+"\t\t"+fSarsa);
			// System.out.println( "0\t" + eval.alpha + "\t" + eval.gamma + "\t" +
			// eval.lambda + "\t" + eval.epsilon + "\t" + eval.randomSeed + "\t" +
			// fSarsa );
		}
		System.out.println("\n====       Averages      ====");
		System.out.println("\t"+sumGRN/maxEval/*+"\t"+sumSarsa/maxEval/**/);
	}

}
