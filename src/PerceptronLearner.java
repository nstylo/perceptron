import java.util.List;

public class PerceptronLearner {

    /* iterations needed for classification */
    private Integer iter;

    /**
     * reads a positive and a negative list of Vectors on which the Perceptron is trained. Outputs
     * a String which contains information about the classification and the amount of iterations it took
     *
     * @param positive      the input-list of positive trainings vectors
     * @param negative      the input-list of negative trainings vectors
     * @param bias          indicates if there is a need for a bias unit
     * @param maxIterations maximal allowed iterations
     * @param queries       the list of Vectors to classify
     * @return a String in the format 'maxIterations', if the learning did not return within the allowed iterations
     * or 'iter (+ or -)(+ or -)*' where iter is the iteration it took to train the Perceptron, + represents a positively
     * classified Vector and - a negatively classified Vector
     */
    public String execute(List<PVector> positive, List<PVector> negative, Boolean bias, Integer maxIterations, List<PVector> queries) {
        // initialize output-string
        String output = "";

        // if there is a need for a bias unit, append the input sets with an extra field with value 1
        if (bias) {
            positive = append(positive);
            negative = append(negative);
            queries = append(queries);
        }

        // learn the Perceptron with the given max iteration count
        PVector weights = perceptronLearning(positive, negative, maxIterations);

        // append the iteration count and return if weights == null, i.e. if iter >= maxIterations
        output = output.concat(iter.toString());
        if (weights == null) {
            return output;
        }

        // for each vector v to classify, do if dot(v, n) <= 0 then classify negatively, else positively
        output = output.concat(" ");
        for (int i = 0; i < queries.size(); i++) {
            PVector v = queries.get(i);
            if (weights.dotProduct(v) <= 0) {
                output = output.concat("-");
            } else {
                output = output.concat("+");
            }
        }

        return output;
    }

    /**
     * This method is reading a list of positive and a list of negative training vectors
     * and outputs the corresponding weights-vector
     *
     * @param positive      the input-list of positive trainings vectors
     * @param negative      the input-list of negative trainings vectors
     * @param maxIterations constrains the amount of iterations for the classification
     * @return the Vector of weights used for classification or null if iter >= maxIterations
     */
    private PVector perceptronLearning(List<PVector> positive, List<PVector> negative, Integer maxIterations) {
        // weights vector -> the output
        PVector weights = new PVector();
        // size of the vectors, we assume all input vectors have the same size
        int n = positive.get(0).size();

        // construct the initial weights vector
        for (int dimensionality = n; dimensionality > 0; dimensionality--) {
            weights = weights.addCoord(1);
        }

        boolean correct;
        iter = 0;
        // while we haven't correctly classified or we have reached maxIterations, adjust weights
        do {
            iter++;
            correct = true;
            for (PVector v : positive) {
                if (weights.dotProduct(v) <= 0) {
                    weights = weights.add(v);
                    correct = false;
                }
            }
            for (PVector v : negative) {
                if (weights.dotProduct(v) > 0) {
                    weights = weights.subtract(v);
                    correct = false;
                }
            }
        } while (!correct || iter >= maxIterations);

        // if we exceed maxIterations, return null
        if (iter >= maxIterations) {
            weights = null;
        }

        return weights;
    }

    private List<PVector> append(List<PVector> vectors) {
        for (int i = 0; i < vectors.size(); i++) {
            PVector v = vectors.get(i).addCoord(1);
            vectors.set(i, v);
        }
        return vectors;
    }
}