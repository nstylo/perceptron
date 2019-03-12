import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 *
 * @author s123188
 * 
 * This class defines a vector to be used for perceptron learning.
 */
public class PVector {
    
    /** 
     * The coordinates of the vector 
     */
    private final List<Integer> coordinates;
    
    /**
     * Creates a PerceptronVector. A copy of the input list is made.
     * @param coordinates the coordinates of the vector as an ArrayList
     */
    public PVector(List<Integer> coordinates){
        this.coordinates = new ArrayList(coordinates);
    }
    
    /**
     * Creates a PVector
     * @param coordinates the coordinates of the vector as an array
     */
    public PVector(Integer ... coordinates){
        this(Arrays.asList(coordinates));
    }
    
    /** Creates a PVector with the given coordinates.
     * 
     * @param coord the coordinates of the vector.
     * @return 
     */
    public static PVector of(Integer ... coord) {
        return new PVector(coord);
    }
    
    /** Creates a  list of PVectors with the given coordinates.
     * <p>
     * Sample usage with output <code>[(1,0), (0,1), (1,1), (0,0)]</code> :
     * <pre>
     * Integer[][] coordinates= { {1,0}, {0,1}, {1,1}, {0,0}};
     * List&lt;PVector&gt; list = PVector.of(coordinates);
     * System.err.println(list);
     * </pre>
     * 
     * @param coord the coordinates of the vector.
     * @return 
     */
    public static List<PVector> listOf(Integer[] ... coord) {
        return Arrays.asList(coord).stream()
                .map(c -> PVector.of(c))
                .collect(Collectors.toList());
    }
    
    /** 
     * Returns a vector of given size and with all coordinates equal to value.
     * @param n    size
     * @param value constant value
     * @return  PVector of size n, with all coordinates equal to value
     */
    public static PVector constant(int n, Integer value) {
        Integer[] coords = new Integer[n];
        Arrays.fill(coords, value);
        return new PVector(coords);
    }
    
    /** Returns dimensionality of this PVector, i.e. the number of coordinates it has.
     * 
     * @return dimensionality 
     */
    public int size() {
        return getVCoords().size();
    }
    
    /**
     * Gets the coordinates of the vector
     * @return a list of the coordinates
     */
    public List<Integer> getVCoords(){
        return coordinates;
    }
    
    /**
     * Adds a coordinate to the vector
     * 
     * @param i the coordinate to be added
     * @return the vector with the added coordinate
     */
    public PVector addCoord(Integer i){
        this.coordinates.add(i);
        return this;
    }
    
    /**
     * Adds two vectors
     * 
     * @param vector the vector to be added
     * @return the result vector
     */
    public PVector add(PVector vector){
        
        List<Integer> coords1 = this.coordinates;
        List<Integer> coords2 = vector.getVCoords();
        
        //check if both vectors have same size
        if(coords1.size() == coords2.size()){
            ArrayList<Integer> newCoords = new ArrayList<>();
            
            //add all coordinates
            for(int i = 0; i < coords1.size(); i++){
                newCoords.add(coords1.get(i) + coords2.get(i));
            }
            
            return new PVector(newCoords);
        }
        else{
            throw new IllegalArgumentException("The vectors must be of equal length!");
        }
    }
    
    /**
     * Subtracts two vectors
     * 
     * @param vector the vector to be subtracted
     * @return the result vector
     */
    public PVector subtract(PVector vector){
        
        List<Integer> coords1 = this.coordinates;
        List<Integer> coords2 = vector.getVCoords();
        
        //check if both vectors have same size
        if(coords1.size() == coords2.size()){
            ArrayList<Integer> newCoords = new ArrayList<>();
            
            //subtract all coordinates
            for(int i = 0; i < coords1.size(); i++){
                newCoords.add(coords1.get(i) - coords2.get(i));
            }
            
            return new PVector(newCoords);
        }
        else{
            throw new IllegalArgumentException("The vectors must be of equal length!");
        }
    }
    
    /**
     * Multiplies two vectors with the dot product
     * 
     * @param vector the vector to be subtracted
     * @return the resulting value
     */
    public Integer dotProduct(PVector vector){
        
        List<Integer> coords1 = this.coordinates;
        List<Integer> coords2 = vector.getVCoords();
        
        //check if both vectors have same size
        if(coords1.size() == coords2.size()){
            Integer result = 0;
            
            //multiply all coordinates and add the results
            for(int i = 0; i < coords1.size(); i++){
                result += coords1.get(i) * coords2.get(i);
            }
            
            return result;
        }
        else{
            throw new IllegalArgumentException("The vectors must be of equal length!");
        }
    }
    
    /**
     * Returns the PVector in string format.
     * Can be used to print PVector information.
     * 
     * @return the PVector in string format as "(coord1,coord2,...)"
     */
    @Override
    public String toString(){
        return coordinates.stream() .map(i->i.toString())
                .collect(Collectors.joining(",", "(", ")"));
    }
    
    public static void main(String[] args) {
        Integer[][] pos = {{1,0}, {0,1},{1,1},{0,0}};
        List<PVector> vectors = PVector.listOf(pos);
        System.err.println(vectors);
    }
}
