import com.dashoptimization.*;

/**
 * Test whether Mosel's jraw driver can initialize an
 * array(integer) of set of integer from Java POJOs.
 *
 * We try multiple approaches:
 *   1. SetEntry POJO with int key + int[] values (nested array approach)
 *   2. FlatSetEntry POJO with int key + int element (flat/sparse approach)
 *   3. Direct int[] binding with noindex for a plain set of integer
 *   4. int[][] binding attempt for array of set
 *
 * Run: javac -cp $XPRESSDIR/lib/xprm.jar *.java
 *      java -cp .:$XPRESSDIR/lib/xprm.jar -Djava.library.path=$XPRESSDIR/lib TestArrayOfSet
 */
public class TestArrayOfSet {

    public static void main(String[] args) {
        XPRM mosel = null;
        try {
            mosel = new XPRM();
            System.out.println("=== Mosel jraw Array(integer) of Set of Integer Test ===\n");

            // -----------------------------------------------------------
            // Test 1: SetEntry POJO approach (key: int, values: int[])
            // -----------------------------------------------------------
            System.out.println("--- Test 1: SetEntry POJO (key + int[] values) ---");
            try {
                SetEntry[] data1 = new SetEntry[] {
                    new SetEntry(1, new int[]{10, 20, 30}),
                    new SetEntry(2, new int[]{40, 50}),
                    new SetEntry(3, new int[]{60})
                };
                mosel.bind("setEntryData", data1);
                XPRMModel model1 = mosel.compile("test_pojo_nested.mos");
                if (model1 != null) {
                    model1.run();
                    System.out.println("  Result: Model ran successfully (exit code: " + model1.getExitCode() + ")");
                    model1.reset();
                } else {
                    // compile returns null if compilation fails - try loading BIM
                    mosel.compile("test_pojo_nested.mos", "test_pojo_nested.bim");
                    XPRMModel m = mosel.load("test_pojo_nested.bim");
                    m.run();
                    System.out.println("  Result: Model ran successfully (exit code: " + m.getExitCode() + ")");
                    m.reset();
                }
            } catch (Exception e) {
                System.out.println("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
            System.out.println();

            // -----------------------------------------------------------
            // Test 2: FlatSetEntry POJO approach (key + single element)
            // -----------------------------------------------------------
            System.out.println("--- Test 2: FlatSetEntry POJO (key + element, flat/sparse) ---");
            try {
                FlatSetEntry[] data2 = new FlatSetEntry[] {
                    new FlatSetEntry(1, 10),
                    new FlatSetEntry(1, 20),
                    new FlatSetEntry(1, 30),
                    new FlatSetEntry(2, 40),
                    new FlatSetEntry(2, 50),
                    new FlatSetEntry(3, 60)
                };
                mosel.bind("flatSetData", data2);
                mosel.compile("test_pojo_flat.mos", "test_pojo_flat.bim");
                XPRMModel model2 = mosel.load("test_pojo_flat.bim");
                model2.run();
                System.out.println("  Result: Model ran successfully (exit code: " + model2.getExitCode() + ")");
                model2.reset();
            } catch (Exception e) {
                System.out.println("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
            System.out.println();

            // -----------------------------------------------------------
            // Test 3: Plain set of integer via noindex int[]
            // -----------------------------------------------------------
            System.out.println("--- Test 3: Plain set of integer from int[] (noindex) ---");
            try {
                int[] plainSet = new int[]{10, 20, 30, 40, 50};
                mosel.bind("plainSetData", plainSet);
                mosel.compile("test_plain_set.mos", "test_plain_set.bim");
                XPRMModel model3 = mosel.load("test_plain_set.bim");
                model3.run();
                System.out.println("  Result: Model ran successfully (exit code: " + model3.getExitCode() + ")");
                model3.reset();
            } catch (Exception e) {
                System.out.println("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
            System.out.println();

            // -----------------------------------------------------------
            // Test 4: int[][] (2D array) attempt for array of sets
            // -----------------------------------------------------------
            System.out.println("--- Test 4: int[][] (2D array) for array of set ---");
            try {
                int[][] data4 = new int[][] {
                    {10, 20, 30},
                    {40, 50},
                    {60}
                };
                mosel.bind("arrayOfArrayData", data4);
                mosel.compile("test_2d_array.mos", "test_2d_array.bim");
                XPRMModel model4 = mosel.load("test_2d_array.bim");
                model4.run();
                System.out.println("  Result: Model ran successfully (exit code: " + model4.getExitCode() + ")");
                model4.reset();
            } catch (Exception e) {
                System.out.println("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }
            System.out.println();

            // -----------------------------------------------------------
            // Test 5: Multiple separate int[] bindings, one per set
            // -----------------------------------------------------------
            System.out.println("--- Test 5: Separate int[] per set member (workaround) ---");
            try {
                int[] set1 = new int[]{10, 20, 30};
                int[] set2 = new int[]{40, 50};
                int[] set3 = new int[]{60};
                mosel.bind("set1", set1);
                mosel.bind("set2", set2);
                mosel.bind("set3", set3);
                mosel.compile("test_separate_sets.mos", "test_separate_sets.bim");
                XPRMModel model5 = mosel.load("test_separate_sets.bim");
                model5.run();
                System.out.println("  Result: Model ran successfully (exit code: " + model5.getExitCode() + ")");
                model5.reset();
            } catch (Exception e) {
                System.out.println("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace(System.out);
            }

            System.out.println("\n=== All tests complete ===");

        } catch (Exception e) {
            System.err.println("Fatal error initializing Mosel: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (mosel != null) {
                mosel.finalize();
            }
        }
    }
}
