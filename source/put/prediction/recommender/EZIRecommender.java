package put.prediction.recommender;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class EZIRecommender {

    public EZIRecommender() {
    }

    public static void main(String[] args) throws Exception {

        DataModel model = new FileDataModel(new File("data/movies.csv"));

        UserSimilarity similarity = new EuclideanDistanceSimilarity(model);

        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model);

        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext(); ) {

            long userID = users.nextLong();

            List<RecommendedItem> recommendations = recommender.recommend(userID, 3);

            for (RecommendedItem recommendation : recommendations) {

                System.out.println(userID + "," + recommendation.getItemID());
            }
        }
        long userID = 943L;
        List<RecommendedItem> recommendations = recommender.recommend(userID, 3);

        for (RecommendedItem recommendation : recommendations) {
            System.out.println(userID + "," + recommendation.getItemID() + "," + recommendation.getValue());
        }
        for (RecommendedItem recommendation : recommendations) {
            for (RecommendedItem recommendation2 : recommendations) {
                System.out.println("sim(" + recommendation.getItemID() + "," + recommendation2.getItemID() + ")" + "=" + ((EuclideanDistanceSimilarity) similarity).itemSimilarity(recommendation.getItemID(), recommendation2.getItemID()));

            }
        }
    }
}


//        sim(258,258)=1.0
//        sim(258,300)=0.43027068256191486
//        sim(258,751)=0.4279887139512474
//        sim(300,258)=0.43027068256191486
//        sim(300,300)=1.0
//        sim(300,751)=0.47688799216733785
//        sim(751,258)=0.4279887139512474
//        sim(751,300)=0.47688799216733785
//        sim(751,751)=1.0