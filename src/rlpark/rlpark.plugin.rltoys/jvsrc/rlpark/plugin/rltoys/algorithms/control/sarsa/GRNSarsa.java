package rlpark.plugin.rltoys.algorithms.control.sarsa;

import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import grn.GRNModel;

public class GRNSarsa extends Sarsa {
	private static final long serialVersionUID = 4783867207342502849L;
	public static double minDelta = -100.0;
	public static double maxDelta = +100.0;
	public double alphaNorm=1.0;

	protected GRNModel grn;
	
	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, GRNModel grn) {
		super(alpha, gamma, lambda, nbFeatures);
		this.grn = grn.copy();
	}
	
	public GRNSarsa(double alpha, double gamma, double lambda, int nbFeatures, Traces prototype, GRNModel grn, double alphaNorm) {
		super(alpha, gamma, lambda, nbFeatures, prototype);
		this.alphaNorm=alphaNorm;
		this.grn = grn.copy();	
	}

	public GRNSarsa(double alpha, double gamma, double lambda, PVector q, Traces prototype, GRNModel grn) {
		super(alpha, gamma, lambda, q, prototype);
		this.grn = grn.copy();	
	}

	@Override
	public double update(RealVector phi_t, RealVector phi_tp1, double r_tp1) {
		// updating parameters
		/*if (delta>maxDelta || delta<minDelta) {
			System.err.println("Delta error : "+delta+"\t"+minDelta+"\t"+maxDelta);
		}*/
//		if (delta!=delta) delta=minDelta;
//		delta=Math.max(delta, minDelta);
//		delta=Math.min(delta, maxDelta);
		grn.proteins.get(0).concentration = Math.min(1.0, Math.max(0.0, (delta-minDelta)/(maxDelta-minDelta)));
		grn.evolve(1);
		this.alpha=grn.proteins.get(1).concentration/(grn.proteins.get(1).concentration+grn.proteins.get(2).concentration+0.001)/alphaNorm;
		this.gamma=grn.proteins.get(3).concentration/(grn.proteins.get(3).concentration+grn.proteins.get(4).concentration+0.001);
		this.lambda=grn.proteins.get(5).concentration/(grn.proteins.get(5).concentration+grn.proteins.get(6).concentration+0.001);
//		System.out.println(grn+" : \t"+delta+"\t"+this.alpha+"\t"+this.gamma+"\t"+this.lambda);
		
		return super.update(phi_t, phi_tp1, r_tp1);
	}
	
	@Override
	protected double initEpisode() {
		//grn.reset();
		return super.initEpisode();
	}

}
