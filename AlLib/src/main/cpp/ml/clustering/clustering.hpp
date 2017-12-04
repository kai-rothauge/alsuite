#ifndef ALML_CLUSTERING_HPP
#define ALML_CLUSTERING_HPP

class Clustering {
public:
	virtual void initialize() = 0;
	virtual void train() = 0;
};

class KMeans : public Clustering {
public:
	KMeans();
	KMeans(int, int, string, int, double, long);

	int getK();
	void setK(int k_);

	int getMaxIterations();
	void setMaxIterations(int maxIterations_);

	string getInitializationMode();
	void setInitializationMode(string initializationMode_);

	int getInitializationSteps();
	void setInitializationSteps(int initializationSteps_);

	double getEpsilon();
	void setEpsilon(double epsilon_);

	long getSeed();
	void setSeed(long seed_);


	void initialize();
	void train();
	void run();

private:
	int k;
	int maxIterations;
	string initializationMode;
	int initializationSteps;
	double epsilon;
	long seed;

	void initialize_random();
	void initialize_parallel();
};

#endif // ALML_CLUSTERING_HPP








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
