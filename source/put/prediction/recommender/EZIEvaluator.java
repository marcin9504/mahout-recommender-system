package put.prediction.recommender;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.*;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

import java.io.File;

public class EZIEvaluator {
    public EZIEvaluator() {
    }

    public static void main(String[] args) throws Exception {

        RandomUtils.useTestSeed();
        DataModel model = new FileDataModel(new File("data/movies.csv"));

        String[] sims = new String[3];
        sims[0] = "EuclideanDistanceSimilarity";
        sims[1] = "PearsonCorrelationSimilarity";
        sims[2] = "TanimotoCoefficientSimilarity";

        UserSimilarity[] similarities = new UserSimilarity[3];
        similarities[0] = new EuclideanDistanceSimilarity(model);
        similarities[1] = new PearsonCorrelationSimilarity(model);
        similarities[2] = new TanimotoCoefficientSimilarity(model);

        String[] models = new String[12];
        UserNeighborhood[] neighborhoods = new UserNeighborhood[12];
        int i = 0;
        for (int j = 0; j < 3; j++) {
            models[i] = "ThresholdUserNeighborhood 0.5" + " " + sims[j];
            neighborhoods[i++] = new ThresholdUserNeighborhood(0.5, similarities[j], model);
            models[i] = "ThresholdUserNeighborhood 0.7" + " " + sims[j];
            neighborhoods[i++] = new ThresholdUserNeighborhood(0.7, similarities[j], model);
            models[i] = "NearestNUserNeighborhood 5" + " " + sims[j];
            neighborhoods[i++] = new NearestNUserNeighborhood(5, similarities[j], model);
            models[i] = "NearestNUserNeighborhood 9" + " " + sims[j];
            neighborhoods[i++] = new NearestNUserNeighborhood(9, similarities[j], model);
        }

        RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
        i = 0;
        for (UserSimilarity similarity : similarities) {

            for (UserNeighborhood neighborhood : neighborhoods) {

                RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
                    @Override
                    public Recommender buildRecommender(DataModel model) {
                        return new GenericUserBasedRecommender(model, neighborhood, similarity);
                    }
                };
                double score = evaluator.evaluate(recommenderBuilder,
                        null,
                        model,
                        0.6,
                        1);
                System.out.println(models[i++]);
                System.out.println(score);
            }
        }

    }
}
//ThresholdUserNeighborhood EuclideanDistanceSimilarity(0.7)