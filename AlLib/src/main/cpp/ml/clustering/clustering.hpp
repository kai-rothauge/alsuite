#ifndef ALLIB__CLUSTERING_HPP
#define ALLIB__CLUSTERING_HPP

#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include <thread>
#include <El.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include "spdlog/spdlog.h"

namespace allib {

class Clustering {
public:
	virtual int initialize() = 0;
	virtual int train() = 0;
};

class KMeans : public Clustering {
public:
	KMeans();
	KMeans(uint32_t, uint32_t, double, string, uint32_t, uint64_t);

	uint32_t get_num_centers();
	void set_num_centers(uint32_t);

	uint32_t get_max_iterations();
	void set_max_iterations(uint32_t);

	std::string get_init_mode();
	void set_init_mode(std::string);

	uint32_t get_init_steps();
	void set_init_steps(uint32_t);

	double get_epsilon();
	void set_epsilon(double);

	uint64_t get_seed();
	void set_seed(uint64_t);

	int initialize();
	int train();
	int run();

private:
	uint32_t num_centers;
	uint32_t max_iterations;					// How many iteration of Lloyd's algorithm to use at most
	double epsilon;							// If all the centers change by Euclidean distance less
											//     than epsilon, then we stop the iterations
	std::string init_mode;					// Number of initialization steps to use in kmeans||
	uint32_t init_steps;						// Which initialization method to use to choose
											//     initial cluster center guesses
	uint64_t seed;							// Random seed used in driver and workers

	int initialize_random();
	int initialize_parallel(DistMatrix const *, MatrixXd const &, uint32_t, MatrixXd &);

	uint32_t update_assignments_and_counts(MatrixXd const &, MatrixXd const &,
	    uint32_t *, std::vector<uint32_t> &, double &);

	int kmeansPP(std::vector<MatrixXd>, std::vector<double>, MatrixXd &);
};

}

#endif // ALLIB__CLUSTERING_HPP








//
//
//
//
///**
// * K-means clustering with support for k-means|| initialization proposed by Bahmani et al.
// *
// * @see <a href="http://dx.doi.org/10.14778/2180912.2180915">Bahmani et al., Scalable k-means++.</a>
// */
//@Since("1.5.0")
//class KMeans @Since("1.5.0") (
//    @Since("1.5.0") override val uid: String)
//  extends Estimator[KMeansModel] with KMeansParams with DefaultParamsWritable {
//
//  setDefault(
//    k -> 2,
//    maxIter -> 20,
//    initMode -> MLlibKMeans.K_MEANS_PARALLEL,
//    initSteps -> 2,
//    tol -> 1e-4)
//
//  @Since("1.5.0")
//  override def copy(extra: ParamMap): KMeans = defaultCopy(extra)
//
//  @Since("1.5.0")
//  def this() = this(Identifiable.randomUID("kmeans"))
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setPredictionCol(value: String): this.type = set(predictionCol, value)
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setK(value: Int): this.type = set(k, value)
//
//  /** @group expertSetParam */
//  @Since("1.5.0")
//  def setInitMode(value: String): this.type = set(initMode, value)
//
//  /** @group expertSetParam */
//  @Since("1.5.0")
//  def setInitSteps(value: Int): this.type = set(initSteps, value)
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setMaxIter(value: Int): this.type = set(maxIter, value)
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setTol(value: Double): this.type = set(tol, value)
//
//  /** @group setParam */
//  @Since("1.5.0")
//  def setSeed(value: Long): this.type = set(seed, value)
//
//  @Since("2.0.0")
//  override def fit(dataset: Dataset[_]): KMeansModel = {
//    transformSchema(dataset.schema, logging = true)
//
//    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
//    val instances: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
//      case Row(point: Vector) => OldVectors.fromML(point)
//    }
//
//    if (handlePersistence) {
//      instances.persist(StorageLevel.MEMORY_AND_DISK)
//    }
//
//    val instr = Instrumentation.create(this, instances)
//    instr.logParams(featuresCol, predictionCol, k, initMode, initSteps, maxIter, seed, tol)
//    val algo = new MLlibKMeans()
//      .setK($(k))
//      .setInitializationMode($(initMode))
//      .setInitializationSteps($(initSteps))
//      .setMaxIterations($(maxIter))
//      .setSeed($(seed))
//      .setEpsilon($(tol))
//    val parentModel = algo.run(instances, Option(instr))
//    val model = copyValues(new KMeansModel(uid, parentModel).setParent(this))
//    val summary = new KMeansSummary(
//      model.transform(dataset), $(predictionCol), $(featuresCol), $(k))
//
//    model.setSummary(Some(summary))
//    instr.logSuccess(model)
//    if (handlePersistence) {
//      instances.unpersist()
//    }
//    model
//  }
//
//  @Since("1.5.0")
//  override def transformSchema(schema: StructType): StructType = {
//    validateAndTransformSchema(schema)
//  }
//}
//
//@Since("1.6.0")
//object KMeans extends DefaultParamsReadable[KMeans] {
//
//  @Since("1.6.0")
//  override def load(path: String): KMeans = super.load(path)
//}
