package Behaviors;

import java.util.ArrayList;
import java.util.Collection;

import Problems.Action;
import Problems.Problem;
import Problems.State;

public abstract class Behavior {
	int nActions;
	int nStates;
	
	public Behavior(int nStates, int nActions) {
		this.nStates=nStates;
		this.nActions=nActions;
	}
	
	public abstract Action chooseAction(State s, ArrayList<Action> possibleActions);
}
