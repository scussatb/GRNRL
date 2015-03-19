package Behaviors;

import java.util.ArrayList;
import java.util.Collection;

import Problems.Action;
import Problems.Problem;
import Problems.State;

public abstract class Behavior {
	
	public abstract Action chooseAction(State s, ArrayList<Action> possibleActions);
}
