package rlpark.plugin.rltoys.math.vector.pool;

import java.util.HashMap;
import java.util.Map;

import rlpark.plugin.rltoys.math.vector.RealVector;

public class VectorPools {
  static private Map<Thread, Map<Class<?>, Map<Integer, ThreadVectorPool>>> pools = new HashMap<Thread, Map<Class<?>, Map<Integer, ThreadVectorPool>>>();

  public static VectorPool pool(RealVector prototype) {
    return pool(prototype, prototype.getDimension());
  }

  public static VectorPool pool(RealVector prototype, int dimension) {
    synchronized (pools) {
      final Thread thread = Thread.currentThread();
      Map<Class<?>, Map<Integer, ThreadVectorPool>> threadPool = pools.get(thread);
      if (threadPool == null) {
        threadPool = new HashMap<Class<?>, Map<Integer, ThreadVectorPool>>();
        pools.put(thread, threadPool);
      }
      Map<Integer, ThreadVectorPool> classPool = threadPool.get(prototype.getClass());
      if (classPool == null) {
        classPool = new HashMap<Integer, ThreadVectorPool>();
        threadPool.put(prototype.getClass(), classPool);
      }
      ThreadVectorPool pool = classPool.get(dimension);
      if (pool == null) {
        pool = new ThreadVectorPool(prototype, dimension);
        classPool.put(dimension, pool);
      }
      pool.allocate();
      return pool;
    }
  }
}
