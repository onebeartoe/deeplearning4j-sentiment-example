
package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import java.io.File;
import java.io.IOException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.DATA_PATH;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.WORD_VECTORS_PATH;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.batchSize;
import static org.deeplearning4j.examples.recurrent.word2vecsentiment.Word2VecSentimentRnnTrain.modelOutfile;
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
    public static void main(String[] args) throws Exception 
    {
        if(args.length != 0)
        {
            
        }
        
        System.out.println("----- Evaluation complete -----");
        
        Word2VecSentimentRnnEvaluate deepLearner = new Word2VecSentimentRnnEvaluate();
        
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelOutfile);                

        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);
        
        deepLearner.evaluate(test, model, truncateReviewsToLength);
        
        System.out.println("----- Evaluation complete -----");
    }
    
    private void evaluate(SentimentExampleIterator test, MultiLayerNetwork model, int truncateReviewsToLength) throws IOException
    {
        //After training: load a single example and generate predictions
        File shortNegativeReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/12100_1.txt"));
        String shortNegativeReview = FileUtils.readFileToString(shortNegativeReviewFile);

        INDArray features = test.loadFeaturesFromString(shortNegativeReview, truncateReviewsToLength);
        INDArray networkOutput = model.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Short negative review: \n" + shortNegativeReview);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Evaluate complete -----");
    }    
}
