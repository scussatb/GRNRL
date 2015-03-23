package evaluators;

import java.sql.Time;
import java.util.Random;
import java.util.Vector;

import javax.print.DocFlavor.STRING;

import rlpark.example.demos.learning.ActorCriticPendulum;
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
	public int nMaxEpisode=30;
	public boolean displayEpisodes = false;
	public Vector<String> problems = new Vector<String>();
	public boolean displayGRN=false;
	public Vector<Double> avgFitPerEpisode = new Vector<Double>();

	// e.rng = new java.util.Random( e.randomSeed );
	public Random rngRL;

	double alpha = 0.1;
	double gamma = 0.9;
	double lambda = 0.1;
	double epsilon = 0.1;
	int randomSeed = (int)System.nanoTime();;

	public NMSarsaEvaluator() {
		numGRNInputs = 3;
		numGRNOutputs = 4;
		nonCacheable=true;
		name = "NMSarsa_uninitialized";
		alpha = 0.15;
		gamma = 1.0;
		lambda = 0.6;
		epsilon = 0.3;
	}
	
	public NMSarsaEvaluator(NMSarsaEvaluator eval) {
		this.fitness=eval.fitness;
		this.nEpisode=eval.nEpisode;
		this.nMaxEpisode=eval.nMaxEpisode;
		this.displayEpisodes=eval.displayEpisodes;
		this.problems=new Vector<String>(eval.problems);
		this.displayGRN=eval.displayGRN;
		this.avgFitPerEpisode=new Vector<Double>(eval.avgFitPerEpisode);
		this.rngRL=eval.rngRL;
		this.alpha=eval.alpha;
		this.gamma=eval.gamma;
		this.lambda=eval.lambda;
		this.epsilon=eval.epsilon;
		this.randomSeed=eval.randomSeed;
	}
	
	public NMSarsaEvaluator clone() {
		return new NMSarsaEvaluator(this);
	}
	
	public NMSarsaEvaluator(boolean displayGRN) {
		this.displayGRN=displayGRN;
		
		numGRNInputs = 4;
		numGRNOutputs = 4;
		nonCacheable=true;
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
			}
		}
		rngRL = new java.util.Random(randomSeed);
/*		System.out.println("alpha = " + alpha);
		System.out.println("gamma = " + gamma);
		System.out.println("lambda = " + lambda);
		System.out.println("epsilon = " + epsilon);
		System.out.println("randomSeed = " + randomSeed);*/
	}

	@Override
	public double evaluate(GRNGenome aGenome) {
		numEvaluations++;
		double fitness = evaluateGRN(buildGRNFromGenome(aGenome), problems, 10, false);
		aGenome.setNewFitness(fitness);
		return fitness;
	}

	public double evaluateGRN(GRNModel grn, Vector<String> problems, int nEval, boolean statistics) {
		double fitness = problems.size()>1?Double.MAX_VALUE:0.0;
		if (problems.contains("MountainCar")) {
			int eval=1;
			double fit;
			double fitMC=evaluateMountainCar(grn, statistics);
			if (fitMC>-400) {
				for (int i=0; i<nEval-1; i++) {
					fit=evaluateMountainCar(grn, statistics);
					fitMC+=fit;
					eval++;
				}
			}
			if (problems.size()>1) {
				fitness=Math.min(fitness, (fitMC/eval+2500)/2350);
//				System.out.print((fitMC/eval+2500)/2350);
			} else {
				fitness+=fitMC;
			}
		}
		if (problems.contains("Maze")) {
			int eval=1;
			double fit;
			double fitMaze=evaluateMaze(grn, statistics);
			if (fitMaze>-100) {
				for (int i=0; i<nEval-1; i++) {
					fit=evaluateMaze(grn, statistics);
					fitMaze+=fit;
					eval++;
				}
			}
			if (problems.size()>1) {
				fitness=Math.min(fitness, (fitMaze*5.0/eval+3000)/2960);
//				System.out.print("\t"+(fitMaze*5.0/eval+3000)/2960+"\t");
			} else {
				fitness+=fitMaze;
			}

		}
		if (problems.contains("ActorCriticPendulum")) {
			int eval=1;
			double fit;
			double fitACP=evaluateActorCriticPendulum(grn, statistics);
			if (fitACP>200) {
				for (int i=0; i<nEval-1; i++) {
					fit=evaluateActorCriticPendulum(grn, statistics);
					fitACP+=fit;
					eval++;
				}
			}
			fitness+=fitACP/eval;
		}
		if (problems.contains("PuddleWorld")) {
			int eval=1;
			double fit;
			double fitPW=evaluatePuddleWorld(grn, statistics);
			if (fitPW>300) {
				for (int i=0; i<nEval-1; i++) {
					fit=evaluatePuddleWorld(grn, statistics);
					fitPW+=fit;
					eval++;
				}
			}
			fitness+=fitPW/eval;
		}
//		 System.out.println(grn+" : "+fitness);
		return fitness;
	}

	public double evaluateMountainCar(GRNModel grn, boolean stats) {

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
		Sarsa sarsa;
		if (grn != null) {
			setSarsaDefaults("MountainCar");
			sarsa = new GRNSarsa(alpha, gamma, lambda,
					toStateAction.vectorSize(), new RTraces(), grn,
					projector.vectorNorm());
//			System.err.println(toStateAction.vectorSize());
			((GRNSarsa) sarsa).minDelta = -33.5031963543928;
			((GRNSarsa) sarsa).maxDelta = 17.841888320550;
			((GRNSarsa) sarsa).displayGRN=displayGRN;
			((GRNSarsa) sarsa).stepMax=200;
		} else {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(),
					new RTraces());
		}

		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
		Zephyr.advertise(clock, this);

		TRStep step = problem.initialize();
		RealVector x_t = null;

		nEpisode = 0;
		fitness = 0;
		nMaxEpisode=25;
		if (stats) {
			avgFitPerEpisode=new Vector<Double>(nMaxEpisode);
		}
		while (clock.tick() && nEpisode < nMaxEpisode) {
			BinaryVector x_tp1 = projector.project(step.o_tp1);
			Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
			x_t = Vectors.bufferedCopy(x_tp1, x_t);
			if (step.isEpisodeEnding() || step.time > 2500) {
				if (displayEpisodes) {
					System.out.println(String.format("%d\t%d", nEpisode, step.time));
//					System.out.println(step.time);
				}
				fitness += step.time;
				if (stats) {
					avgFitPerEpisode.add(new Double(step.time));
				}
				step = problem.initialize();
				x_t = null;
				nEpisode++;
			} else {
				step = problem.step(action);
			}
		}
		return -fitness / nMaxEpisode;

	}

	public double evaluateMaze(GRNModel grn, boolean stats) {
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
			((GRNSarsa) sarsa).displayGRN=displayGRN;
			((GRNSarsa) sarsa).stepMax=100;
		}
		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, projector);
		mazeValueFunction = new MazeValueFunction(problem, sarsa,
				toStateAction, acting);
		Zephyr.advertise(clock, this);

		nEpisode = 0;
		nMaxEpisode=30;
		Runner runner = new Runner(problem, agent, nMaxEpisode, 3000);
		fitness = 0;
		if (stats) {
			avgFitPerEpisode=new Vector<Double>(nMaxEpisode);
		}
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			public boolean stats = true;
			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%d",
							eventInfo.nbEpisodeDone, eventInfo.step.time));
				nEpisode++;
				fitness += eventInfo.step.time;
				if (stats) {
					avgFitPerEpisode.add(new Double(eventInfo.step.time));
				}
			}
		});
		while (clock.tick() && nEpisode < nMaxEpisode) {
			runner.step();
			occupancy.addToSelf(agent.lastState());
		}

		return -fitness / nMaxEpisode;
	}

	public double evaluatePuddleWorld(GRNModel grn, boolean stats) {
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
			((GRNSarsa) sarsa).displayGRN=displayGRN;
			((GRNSarsa) sarsa).stepMax=300;
		}
		Policy acting = new EpsilonGreedy(rngRL, world.actions(),
				toStateAction, sarsa, epsilon);
		ControlLearner control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, tileCoders);
		valueFunction = new ValueFunction2D(tileCoders, world, sarsa);
		
		nMaxEpisode=40;
		nEpisode = 0;
		fitness = 0;
		runner = new Runner(world, agent, nMaxEpisode, 500);
		if (stats) {
			avgFitPerEpisode=new Vector<Double>(nMaxEpisode);
		}
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			public boolean stats = true;

			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%f",
							eventInfo.nbEpisodeDone, eventInfo.episodeReward));
				nEpisode++;
				fitness += eventInfo.episodeReward;
				if (stats) {
					avgFitPerEpisode.add(new Double(eventInfo.episodeReward));
				}
			}
		});

		Zephyr.advertise(clock, this);

		while (clock.tick() && nEpisode < nMaxEpisode) {
			runner.step();
		}

		return fitness / nMaxEpisode;

	}

	public double evaluateActorCriticPendulum(GRNModel grn, boolean stats) {
		FunctionProjected2D valueFunction;
		double reward;
		SwingPendulum problem;
		Clock clock = new Clock("ActorCriticPendulum");
		LearnerAgentFA agent;
		Runner runner;

		Random random = rngRL;
		problem = new SwingPendulum(null, false);
		TileCodersNoHashing tileCoders = new TileCodersNoHashing(
				problem.getObservationRanges());
		((AbstractPartitionFactory) tileCoders.discretizerFactory()).setRandom(
				random, .2);
		tileCoders.addFullTilings(10, 10);

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
			((GRNSarsa) sarsa).displayGRN=displayGRN;
			((GRNSarsa) sarsa).stepMax=600;
		}

		Policy acting = new EpsilonGreedy(rngRL, problem.actions(),
				toStateAction, sarsa, epsilon);
		ControlLearner control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, tileCoders);
		valueFunction = new ValueFunction2D(tileCoders, problem, sarsa);
		nEpisode = 0;
		nMaxEpisode=30;
		fitness = 0;
		runner = new Runner(problem, agent, nMaxEpisode, 1000);
		if (stats) {
			avgFitPerEpisode=new Vector<Double>(nMaxEpisode);
		}
		runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
			public boolean stats = true;

			@Override
			public void listen(RunnerEvent eventInfo) {
				if (displayEpisodes)
					System.out.println(String.format("%d\t%f",
							eventInfo.nbEpisodeDone, eventInfo.episodeReward));
				nEpisode++;
				fitness += eventInfo.episodeReward;
				if (stats) {
					avgFitPerEpisode.add(new Double(eventInfo.episodeReward));
				}

			}
		});

		Zephyr.advertise(clock, this);

		while (clock.tick() && nEpisode < nMaxEpisode) {
			runner.step();
		}

		return fitness / nMaxEpisode;

	}

	public static void main(String[] args) throws Exception {
		GRNModel grnMAZE = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_Maze/run_2784148435803090/grn_492_-54.70933333333333.grn");	
		GRNModel grnGeneric = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_Maze-MountainCar/run_3536341606441451/grn_147_1.9319125531914894.grn");
/*MC	GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_MountainCar/run_2784175489273741/grn_105_-167.6384.grn"); /**/
/*MZ	GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_Maze/run_2784148435803090/grn_492_-54.70933333333333.grn"); /**/
/*PW*/	GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_PuddleWorld/run_3274425346046829/grn_120_718.2049999999942.grn"); /**/
/*ACP	GRNModel grn = GRNModel.loadFromFile("/Users/cussat/Recherche/Projets/Neuromodulation/GRNRL/Results/run_2015-01-23/NMSarsa_ActorCriticPendulum/run_76392139137808/grn_38_425.5604306348474.grn"); /**/

		
		float sumGRN=0;
		float sumSarsa=0;
		float sumGRNMaze=0;
		float sumGRNGeneric=0;
		
		int maxEval=1;
		Vector<Double> avgFitGRN=new Vector<Double>();
		Vector<Double> avgFitSARSA=new Vector<Double>();
		Vector<Double> avgFitGRNMAZE=new Vector<Double>();
		Vector<Double> avgFitGRNGeneric=new Vector<Double>();
		
		for (int i=0; i<maxEval; i++) {
			NMSarsaEvaluator eval = new NMSarsaEvaluator(/*false/**/maxEval==1/**/);
			eval.displayEpisodes = false;
			

//			eval.problems.add("MountainCar");
//			eval.alpha =  0.07142857142854546;
//			eval.gamma = 1.0;
//			eval.lambda = 0.928571428571;
//			eval.epsilon = 0.01;
//			eval.problems.add("Maze");
//			eval.alpha =  1.0;
//			eval.gamma = 0.928571428571;
//			eval.lambda = 0.928571428571;
//			eval.epsilon = 0.01;
//			eval.problems.add("PuddleWorld");
			eval.alpha =   0.0571428571429;
			eval.gamma = 0.928571428571;
			eval.lambda = 0.5;
			eval.epsilon = 0.01;
			eval.problems.add("ActorCriticPendulum");
//			eval.alpha =   0.05;
//			eval.gamma = 0.928571428571;
//			eval.lambda = 0.785714285714;
//			eval.epsilon = 0.01;
		
			
			eval.rngRL = new java.util.Random(i*1000);

//			 System.out.println("====   Evaluating SARSA  ====");
			double fSarsa=0;
//			fSarsa = eval.evaluateGRN(null, eval.problems, 1, true);
//			for (int ep=0;ep<eval.avgFitPerEpisode.size(); ep++) {
//				if (avgFitSARSA.size()<=ep) {
//					avgFitSARSA.add(eval.avgFitPerEpisode.get(ep));
//				} else {
//					avgFitSARSA.set(ep, avgFitSARSA.get(ep)+eval.avgFitPerEpisode.get(ep));
//				}
//			}

//			 System.out.println("\n==== Evaluating GRNSARSA ====");
			double fGRN = 0.0;
//			fGRN = eval.evaluateGRN(grn, eval.problems, 1, true);
//			for (int ep=0;ep<eval.avgFitPerEpisode.size(); ep++) {
//				if (avgFitGRN.size()<=ep) {
//					avgFitGRN.add(eval.avgFitPerEpisode.get(ep));
//				} else {
//					avgFitGRN.set(ep, avgFitGRN.get(ep)+eval.avgFitPerEpisode.get(ep));
//				}
//			}
			
//			 System.out.println("\n==== Evaluating GRNSARSA_MAZE ====");
			double fGRNMaze=0.0;
//			fGRNMaze = eval.evaluateGRN(grnMAZE, eval.problems, 1, true);
//			for (int ep=0;ep<eval.avgFitPerEpisode.size(); ep++) {
//				if (avgFitGRNMAZE.size()<=ep) {
//					avgFitGRNMAZE.add(eval.avgFitPerEpisode.get(ep));
//				} else {
//					avgFitGRNMAZE.set(ep, avgFitGRNMAZE.get(ep)+eval.avgFitPerEpisode.get(ep));
//				}
//			}
			
			double fGRNGeneric=0.0;
			fGRNGeneric = eval.evaluateGRN(grnGeneric, eval.problems, 1, true);
			for (int ep=0;ep<eval.avgFitPerEpisode.size(); ep++) {
				if (avgFitGRNGeneric.size()<=ep) {
					avgFitGRNGeneric.add(eval.avgFitPerEpisode.get(ep));
				} else {
					avgFitGRNGeneric.set(ep, avgFitGRNGeneric.get(ep)+eval.avgFitPerEpisode.get(ep));
				}
			}
			
			sumGRN+=fGRN;
			sumSarsa+=fSarsa;
			sumGRNMaze+=fGRNMaze;
			sumGRNGeneric+=fGRNGeneric;
			
			System.out.println(/*i+"\t"+fGRN+"\t"+fSarsa+"\t"+fGRNMaze+"\t"+*/fGRNGeneric);
		}
		System.out.println("\n====       Averages      ====");
		System.out.println("\t"+sumGRNGeneric/maxEval);//+"\t"+sumSarsa/maxEval+"\t"+sumGRNMaze/maxEval+"\t"+sumGRNGeneric/maxEval);

		for (int i=0; i<Math.min(avgFitGRNGeneric.size(), 100/*avgFitSARSA.size()*/); i++) {
			System.out.println(/*i+"\t"+*/avgFitGRNGeneric.get(i)/maxEval);//+"\t"+avgFitSARSA.get(i)/maxEval+"\t"+avgFitGRNMAZE.get(i)/maxEval+"\t"+*/avgFitGRNGeneric.get(i)/maxEval);
		}
	}

}
