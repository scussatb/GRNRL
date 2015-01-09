package rlpark.plugin.rltoys.algorithms.control.sarsa;

import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import grn.GRNModel;

public class GRNSarsa extends Sarsa {
	private static final long serialVersionUID = 4783867207342502849L;

	protected GRNModel grn;
	
	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, GRNModel grn) {
		super(alpha, gamma, lambda, nbFeatures);
		this.grn = grn.copy();
	}
	
	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, Traces prototype, GRNModel grn) {
		super(alpha, gamma, lambda, nbFeatures, prototype);
		this.grn = grn.copy();	
	}

	public GRNSarsa(double alpha, double gamma, double lambda, PVector q, Traces prototype, GRNModel grn) {
		super(alpha, gamma, lambda, q, prototype);
		this.grn = grn.copy();	
	}

	@Override
	public double update(RealVector phi_t, RealVector phi_tp1, double r_tp1) {
		// updating parameters
//		grn.proteins.get(0).concentration = this.delta;
		grn.proteins.get(0).concentration = Math.min(1.0, Math.max(this.delta+1, 0.0));
		grn.evolve(1);
		this.alpha=grn.proteins.get(1).concentration;
		this.gamma=grn.proteins.get(2).concentration;
		this.lambda=grn.proteins.get(3).concentration;
		//System.out.println(this.delta+"\t"+this.alpha+"\t"+this.gamma+"\t"+this.lambda);
		
		return super.update(phi_t, phi_tp1, r_tp1);
	}
	
	@Override
	protected double initEpisode() {
		//grn.reset();
		return super.initEpisode();
	}

}
