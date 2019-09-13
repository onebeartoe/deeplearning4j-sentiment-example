
package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import java.io.File;
import java.io.IOException;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.DATA_PATH;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.WORD_VECTORS_PATH;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.batchSize;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.truncateReviewsToLength;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * This class evaluates a trained neural network model.
 * 
 * @author Roberto Marquez
 */
public class Word2VecSentimentRnnEvaluate
{
//TODO: clean up this main method so that it uses Apache CLI and uses some OO concepts    
    public static void main(String[] args) throws Exception 
    {
        System.out.println("----- Evaluation initializing -----");
        
        String trainedModelPath = null;
        
        if(args.length != 0)
        {
            trainedModelPath = args[0];
            
            System.out.println("using command line arguement for trained model path: " + trainedModelPath);
        }
        else
        {
            System.out.println("please provide the path the trained model as an command line argument.");
            System.exit(1);
        }
        
        if(args.length > 1)
        {
//TODO: eek!  stop using this class member this way.  it hurts my soul!            
            WORD_VECTORS_PATH = args[1];
        }
        else
        {
            System.out.println("please provide the word vectors path as the second argument to the command line.");
            System.exit(1);
        }
        
        System.out.println("----- Evaluation starting -----");
        
        Word2VecSentimentRnnEvaluate deepLearner = new Word2VecSentimentRnnEvaluate();
        
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(trainedModelPath);                

        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);
        
        // expected bad review
        String shortNegativeReview = "Boy, did that movie suck. It was like a bad version of my least favorite cartoon.";        
        deepLearner.evaluate(test, model, truncateReviewsToLength, shortNegativeReview);
        
        // another expected bad review
        String secondBadReview = "Homer - Yeah Moe that team sure did suck last night. They just plain sucked! I've seen teams suck before, but they were the suckiest bunch of sucks that ever sucked.";                
        deepLearner.evaluate(test, model, truncateReviewsToLength, secondBadReview);
        
        // a good review follows (hopefully)
        String goodReview = "Boy, did I sure enjoy that movie.  It was great!";        
        deepLearner.evaluate(test, model, truncateReviewsToLength, goodReview);
        
        System.out.println("----- Evaluation complete -----");
    }

    /**
     * After training: load a single example and generate predictions
     * @param test
     * @param model
     * @param truncateReviewsToLength
     * @param review
     * @throws IOException 
     */
    private void evaluate(SentimentExampleIterator test, MultiLayerNetwork model, 
                            int truncateReviewsToLength, String review) throws IOException
    {
        INDArray features = test.loadFeaturesFromString(review, truncateReviewsToLength);
        INDArray networkOutput = model.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Review: \n" + review);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Evaluate complete -----");
    }    
}
