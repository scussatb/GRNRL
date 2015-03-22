package experiments;

import learning.SARSA;
import Behaviors.RandomBehavior;
import Problems.Maze;

public class SARSAMaze {
	
	public static void main(String[] args) {		
		Maze maze=new Maze();
		SARSA behavior = new SARSA( maze.numStates(), maze.numActions(), System.currentTimeMillis());
		behavior.alpha = 0.5;
		behavior.gamma = 0.9;
		behavior.lambda = 0.9;
		behavior.verbosity = 1;
		
		behavior.training( maze, new String[]{"epsilonGreedy", "0.1"}, 500, 1000);
		
		behavior.saveMatrixtoCSV( behavior.q, "SarsaMazeQ.csv" );
		behavior.saveMatrixtoCSV( behavior.e, "SarsaMazeE.csv" );
		
	}

}
