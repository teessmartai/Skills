/**
 * Alternative POJO: flat representation where each row maps
 * one array key to one set member (repeated keys for multi-element sets).
 *
 * This matches the standard jraw pattern of POJO-array with
 * index fields + value field per row.
 */
public class FlatSetEntry {
    /** The array index key */
    public int key;
    /** A single element of the set for this key */
    public int element;

    public FlatSetEntry() {}

    public FlatSetEntry(int key, int element) {
        this.key = key;
        this.element = element;
    }
}
