package Problems;

import java.util.ArrayList;

import Behaviors.Behavior;

public abstract class Problem {
	public int curStep;
	public ArrayList<Action> possibleActions;
	
	public Problem() {
	}
	
	public abstract double run(Behavior behavior, int maxSteps);
	public abstract boolean isSolved();
	public abstract int getNStates();
	public abstract int getNActions();
	
	public abstract double getReward(State s);
	
	public String displayCurrentState() {
		return "";
	}

	//public abstract void stepForward(Behavior behavior);
	public abstract void stepForward(Action action);	
	
	
	public abstract int numActions();
	public abstract int numStates();
	public abstract void reset();
	public abstract State getCurrentState();
}
