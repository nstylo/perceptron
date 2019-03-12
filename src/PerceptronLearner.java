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

        // if there is a need for a bias unit, appendCoord the input sets with an extra field with value 1
        if (bias) {
            appendCoord(positive, 1);
            appendCoord(negative, 1);
            appendCoord(queries, 1);
        }

        // learn the Perceptron with the given max iteration count
        PVector weights = perceptronLearning(positive, negative, maxIterations);

        // appendCoord the iteration count and return if weights == null, i.e. if iter >= maxIterations
        output += iter;
        if (weights == null) {
            return output;
        }

        // for each vector v to classify, do if dot(v, n) <= 0 then classify negatively, else positively
        output += " ";
        for (PVector v : queries) {
            if (weights.dotProduct(v) <= 0) {
                output += "-";
            } else {
                output += "+";
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
        // size of the vectors, we assume all input vectors have the same size
        int n = positive.get(0).size();

        // weights vector -> the output
        PVector weights = PVector.constant(n, 1);

        boolean correct = false;
        iter = 0;
        // while we haven't correctly classified and we have reached maxIterations, adjust weights
        while (!correct && iter < maxIterations) {
            iter++;
            correct = true;

            // classify positive
            for (PVector v : positive) {
                if (weights.dotProduct(v) <= 0) {
                    weights = weights.add(v);
                    correct = false;
                }
            }

            // classify negative
            for (PVector v : negative) {
                if (weights.dotProduct(v) > 0) {
                    weights = weights.subtract(v);
                    correct = false;
                }
            }
        }

        // if we meet maxIterations, return null
        if (iter >= maxIterations) {
            return null;
        }

        return weights;
    }

    /**
     * appends vectors in a list with another coordinate with a given value
     *
     * @param vectors list of vectors
     * @param val     value to assign the new coordinate
     * @return the input list with appended vectors
     */
    private void appendCoord(List<PVector> vectors, int val) {
        // for each of the vectors, add a new coordinate with value
        for (int i = 0; i < vectors.size(); i++) {
            PVector v = vectors.get(i).addCoord(val);
            vectors.set(i, v);
        }
    }
}