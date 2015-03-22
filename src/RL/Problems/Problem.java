package Problems;

import Behaviors.Behavior;

public abstract class Problem {
	public Problem() {
	}
	
	public abstract double run(Behavior behavior, int maxSteps);
	public abstract boolean isSolved();
	
	public abstract double getReward(State s);
	
	public String displayCurrentState() {
		return "";
	}

	public abstract void stepForward(Behavior behavior);		
}
