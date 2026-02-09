# Mosel jraw: Can you initialize `array(integer) of set of integer`?

## Question

Can the Mosel `jraw` IO driver initialize an `array(integer) of set of integer`
from a Java POJO that has an `int` key field and an `int[]` values field?

## TL;DR — Expected Answer: **No (not directly)**

The `jraw` driver documentation states:

> Given the restrictions of the raw format (each 'file' only contains a single
> data array, **composite Mosel data structures are not supported**), it is
> recommended to give preference to the more flexible binary format defined
> by the `bin` driver.

An `array(integer) of set of integer` is a **composite** data structure
(an array whose values are themselves sets), so it falls outside what
`jraw`/`raw` supports.

## Test Approaches

This project tests 5 different strategies to see which ones work or fail:

| Test | Java Binding | Mosel Target | Expected Result |
|------|-------------|--------------|-----------------|
| 1 | `SetEntry[]` (key + int[]) | `array(range) of set of integer` | **FAIL** — composite type not supported by jraw |
| 2 | `FlatSetEntry[]` (key + element) | `array(range) of set of integer` | **FAIL** — still a composite target type |
| 3 | `int[]` with noindex | `set of integer` | **PASS** — simple set, basic type |
| 4 | `int[][]` with noindex | `array(range) of set of integer` | **FAIL** — nested arrays not supported |
| 5 | Separate `int[]` per set | `set of integer` x3, assembled after | **PASS** — workaround using simple types |

### Test 1: SetEntry POJO (key + int[] values)
- Java: `SetEntry { int key; int[] values; }` array bound as `"setEntryData"`
- Mosel: `S as "setEntryData(key,values)"` targeting `array(KEYS) of set of integer`
- **Expected**: Fails because jraw cannot map an `int[]` field to a set-of-integer value

### Test 2: FlatSetEntry POJO (key + element, flat/sparse)
- Java: `FlatSetEntry { int key; int element; }` with repeated keys
- Mosel: `S as "flatSetData(key,element)"` targeting `array(range) of set of integer`
- **Expected**: Fails because the target type is composite, even though the POJO is flat

### Test 3: Plain set of integer (baseline)
- Java: `int[]` bound with noindex
- Mosel: `S as "noindex,plainSetData"` targeting `set of integer`
- **Expected**: Succeeds — this is a basic type, well-documented

### Test 4: int[][] (2D array)
- Java: `int[][]` bound with noindex
- Mosel: `S as "noindex,arrayOfArrayData"` targeting `array(1..3) of set of integer`
- **Expected**: Fails — jraw expects flat Java objects matching basic Mosel types

### Test 5: Separate int[] per set (workaround)
- Java: Three separate `int[]` bindings (`set1`, `set2`, `set3`)
- Mosel: Load into temporary `set of integer` vars, then assign to array slots
- **Expected**: Succeeds — each individual transfer is a simple type

## Workarounds

If you need `array(integer) of set of integer` populated from Java:

1. **Separate bindings** (Test 5): Bind each set as its own `int[]`, load into
   temp sets, assemble into the array. Works but doesn't scale.

2. **Use `XPRMInitializationFrom` callback interface**: Implement the Java
   interface to programmatically feed data into Mosel's initialization context.
   Most flexible but more complex.

3. **Use `bin` driver instead of `jraw`**: The `bin` driver supports composite
   data structures. Write data in Mosel's binary format from Java.

4. **Use a data file** (CSV/dat): Write the data to a file from Java, then
   read it with `initializations from "data.dat"` in Mosel. The text format
   supports arbitrary composite structures.

5. **Flatten to simple arrays**: Instead of `array(integer) of set of integer`,
   redesign the Mosel model to use `array(integer, integer) of integer` or
   similar flat representation that jraw can handle.

## How to Run

```bash
# Requires FICO Xpress with Mosel installed
export XPRESSDIR=/path/to/xpressmp
./build_and_run.sh
```

## File Structure

```
src/
├── SetEntry.java           # POJO: int key + int[] values
├── FlatSetEntry.java       # POJO: int key + int element (flat)
├── TestArrayOfSet.java     # Main driver running all 5 tests
├── test_pojo_nested.mos    # Test 1: SetEntry POJO approach
├── test_pojo_flat.mos      # Test 2: FlatSetEntry flat approach
├── test_plain_set.mos      # Test 3: Plain set of integer baseline
├── test_2d_array.mos       # Test 4: int[][] 2D array attempt
└── test_separate_sets.mos  # Test 5: Separate bindings workaround
```

## References

- [mmjava I/O drivers](https://www.fico.com/fico-xpress-optimization/docs/dms2019-04/mosel/mosel_lang/dhtml/mmjava_sec_secjavaio.html)
- [raw driver documentation](https://www.fico.com/fico-xpress-optimization/docs/dms2018-04/mosel/mosel_io/dhtml/secio2_sec_secioraw.html)
- [Mosel User Guide — Java](https://www.fico.com/fico-xpress-optimization/docs/dms2018-04/mosel/UG/dhtml/moselugC2_sec_secc2java.html)
- [Arrays initialization](https://www.fico.com/fico-xpress-optimization/docs/dms2020-03/mosel/UG/dhtml/moselugB2_sec_secB2initarr.html)
