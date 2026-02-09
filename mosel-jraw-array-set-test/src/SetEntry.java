/**
 * POJO representing one entry for an array(integer) of set of integer.
 *
 * The idea: each entry maps an array key (integer) to an element
 * that belongs to that key's set.
 *
 * For jraw with indexed data, the first field(s) are treated as indices
 * and the last field as the value. But "set of integer" is a composite
 * type — the question is whether jraw can handle this.
 */
public class SetEntry {
    /** The array index key */
    public int key;
    /** The set elements for this key */
    public int[] values;

    public SetEntry() {}

    public SetEntry(int key, int[] values) {
        this.key = key;
        this.values = values;
    }
}
