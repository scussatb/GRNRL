package Problems;

import Behaviors.Behavior;


public abstract class Problem {
	protected Behavior behavior;

	public Problem(Behavior b) {
		this.behavior = b;
	}
	
	public abstract double run(int maxSteps);
	public abstract void stepForward();
	public abstract boolean isSolved();
	public abstract int getNStates();
	public abstract int getNActions();
	
	public abstract double getReward(State s);
	
	public String displayCurrentState() {
		return "";
	}
}
