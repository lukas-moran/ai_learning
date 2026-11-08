# Readiness: Python, Small Tests, and Mathematical Language

Use this chapter before W01 or whenever syntax hides the mathematical idea. Progress is based on explaining and changing a working example, not on pages read.

## Concepts

A variable name refers to an object. If a=[1,2] and b=a, appending to b also changes what a observes. A shallow copy creates a new outer list, but nested objects can still be shared.

A function should state its input contract and return its result. print displays information but does not replace return. For a mean function, decide what an empty sequence means before writing division; this route uses None for an empty sequence.

Branches are checked in order. A loop processes each item; a return inside that loop exits the entire function. Use tiny inputs where every intermediate total can be predicted.

Lists preserve order, dictionaries map keys, and sets store distinct members. Read files with a context manager and explicit encoding. Catch expected exceptions where you can provide recovery.

A derivative is a local rate of change. For L(w)=(2w-3)^2, the chain rule gives 4(2w-3); at w=1 it is -4, so a sufficiently small positive change in w reduces the loss.

Conditional probability restricts the reference population: P(A|B)=P(A and B)/P(B). If X is 0 or 2 with equal probability, E[X]=1 and Var(X)=1 even though X never equals its expectation.

A dot product multiplies corresponding coordinates and adds them. For x=[1,2] and y=[3,4], it is 11, while elementwise multiplication gives [3,8].

## Detailed study

## Following a program: names, objects and contracts

Suppose a sensor supplies the strings "12", "15" and "bad". Before calculating an average, three different questions arise: what type of object do we have, which values are valid, and what should happen if no valid measurements remain? Python will not choose the scientific interpretation for us. Converting "12" with float produces a number; converting "bad" raises ValueError. Replacing every failed conversion with zero silently invents measurements. A small program becomes easier to trust when its input and output contracts are stated before the loop.

A name is a reference to an object. With a=[1,2] and b=a, both names refer to one list. By contrast, b=a.copy() creates a new outer list. If a contains nested lists, those inner objects are still shared after a shallow copy. Draw two names pointing to the same box before memorizing the word mutable（可变的）. This model later explains why changing an array view can change its source.

Separate calculation from display. A function returning a number can be tested, composed and reused in a notebook. A function that only prints that number returns None unless another return is executed. Decide whether empty input is an error or a supported case. Here safe_mean returns None for empty input; other interfaces could reasonably raise ValueError. The important property is a consistent, documented contract.

## From loops to mathematical notation

Read a sum as a program with an accumulator. In sum of x_i squared, start total at zero, visit each observation, square it, and add the result. A return inside the loop stops after the first observation. The mathematical summation sign hides this control flow, so a three-element hand trace is often more useful than a large random test.

A derivative describes change per unit input near one point. For f(w)=(w-2)^2, changing w from 0 to 0.01 changes f from 4 to 3.9601. The quotient is -3.99, close to f'(0)=-4. The derivative is a local prediction, not the exact change for an arbitrary jump. With several parameters, a partial derivative changes one coordinate while holding the others fixed.

Probability requires a reference population. If 20 of 100 people exercise and 8 of those 20 cycle, P(cycle given exercise)=8/20. This differs from 8/100. Expectation is a weighted average over outcomes, not the most likely outcome and not necessarily a possible outcome. These distinctions are the minimum mathematical vocabulary needed for loss functions and classifiers.

## The argument in mathematical notation

### A mean and a local derivative

$
\bar x=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad f'(w)=\lim_{h\to0}\frac{f(w+h)-f(w)}{h}
$

The mean needs a nonzero observation count; the derivative is a limit of a local change ratio.

## A probability bridge: distinguish a value, a sample and a distribution

Let a random variable X be 0 with probability one half and 2 with probability one half. Its expectation is 1, even though 1 is never observed. Its variance is the expected squared distance from 1: one half times 1 plus one half times 1, giving 1. A sample [0,0,2] has mean 2/3. This is not a contradiction: the distribution describes possible samples, while one sample is a realized collection.

Now average two independent draws. Possible means are 0,1,2 with probabilities 1/4,1/2,1/4. The mean of that distribution is still 1, but its variance is 1/2. Averaging reduces variation here because independent deviations partly cancel. If the two draws are forced to be identical instead, their average has variance 1. Dependence matters. This small enumeration prepares the later discussion of averaging models and repeated measurements.

When you report a sample mean, distinguish what was observed from the population quantity it estimates. A larger independent sample usually stabilizes that estimate under suitable conditions; copying existing observations does not create new independent information. Keep that distinction in mind when constructing a training dataset or repeating code.

## Trace a valid-data average without losing information

1. Input is ["12", "15", "bad", "0"]. Start valid=[] and rejected=0. Convert the first two entries to 12.0 and 15.0.

2. Conversion fails only for "bad". Increment rejected, retaining the distinction between missing information and a measured zero. Convert the last entry to 0.0.

3. There are three accepted values. Their sum is 27 and their average is 9. Dividing by the original four entries would give 6.75 and would answer a different question.

4. For an entirely invalid input, report no average and the number rejected. Do not divide by zero or report an invented zero average. The report should expose both the statistic and its denominator.

## Build tests around behavior rather than syntax

Start with an ordinary input, then an empty input, one valid observation, an actual zero and an invalid string. The expected output should be calculated independently. An assertion such as mean(xs)==mean(xs) cannot detect an incorrect mean implementation. Catch ValueError around conversion, where recovery is possible; avoid wrapping the entire program in a broad exception handler that hides programming errors.

## An inspectable conversion pipeline

```python
def summarize(raw):
    values = []
    rejected = 0
    for item in raw:
        try:
            value = float(item)
        except ValueError:
            rejected += 1
            continue
        values.append(value)
    average = sum(values) / len(values) if values else None
    return average, len(values), rejected

assert summarize(["12", "15", "bad", "0"]) == (9.0, 3, 1)
assert summarize([]) == (None, 0, 0)
assert summarize(["bad"]) == (None, 0, 1)
assert summarize(["0"]) == (0.0, 1, 0)
print(summarize(["12", "15", "bad", "0"]))
```

## Debug by finding the first violated expectation

Write the expected list after every iteration before running the function. If the final average is wrong, inspect accepted values before changing division. If the accepted list is already wrong, inspect conversion or the branch. This narrows the search: an error at the end may originate much earlier. Then replace one input at a time. Changing both the data and the algorithm prevents you from identifying which change fixed the result.

The official Python tutorial develops language behavior; short exercise sites provide practice. Neither supplies the measurement policy in this example. That policy belongs to the application. When moving to statistical learning, preserve this distinction: a library determines what a call does, while the problem determines whether it is the right call.

## Further practice with worked solutions

### Why is a denominator part of a statistical result?

Solution: The mean of valid measurements uses the valid count. Dividing by all input records implicitly assigns a different treatment to rejected records. State which observations contributed.

### Change the contract so invalid text raises an error. What changes?

Solution: Remove the conversion recovery or raise a descriptive ValueError. The all-invalid input now fails visibly instead of producing a missing average. Update tests to assert that behavior; neither contract is universally correct.

## Worked example

1. sum(range(1,101)) is 5050 because 101 is excluded.
2. mean([2,4,6]) accumulates 12 and divides by 3 after the loop.
3. For numeric input, convert text, reject invalid numbers, and check a zero denominator before dividing.
4. Test the highest score threshold first, or use nonoverlapping conditions.

## Comparing the sources



## Implementation

Write square_sum(xs), safe_mean(xs), and score_category(score). Define boundaries and empty-input behavior.

Test empty, one-element and ordinary inputs.

Read a short UTF-8 file, count words with a dictionary, and explain why a set cannot retain frequencies.

## Controlled experiment

Predict each output before running it. Change one input at a time.

Compare b=a with b=a.copy(), including a nested-list example.

Use LeetCode, Exercism or Codewars to practice one demonstrated weakness; an accepted copied answer is not independent evidence.

## Self-check

1. What differs between return and print?

Answer: return sends a value to the caller; print displays text.

2. What does safe_mean([]) return here?

Answer: None.

3. What is sum(range(1,101))?

Answer: 5050.

4. Why can b.append change a?

Answer: Both names refer to the same mutable list.

5. What permits W01?

Answer: The three functions, a dot product and a simple derivative can be explained and changed independently.

## One-page recap

Predict, run, inspect and explain. Locate the first incorrect assumption when results surprise you.

Enter W01 when this gate is met.

## Terms

- function (函数): A reusable computation with inputs and a returned result.
- mutable (可变的): Able to be modified without replacing the object.
- exception (异常): A runtime signal that normal execution cannot continue as written.
- conditional probability (条件概率): A probability calculated within outcomes satisfying a condition.
- expectation (期望): A probability-weighted average of possible values.

## Sources

- [Python Tutorial](https://docs.python.org/3/tutorial/): Tutorial: control flow, data structures, files and exceptions
- [Exercism Python](https://exercism.org/tracks/python): Python track
- [LeetCode problem set](https://leetcode.com/problemset/): Problem set and existing ability assessment
- [Mathematics for Machine Learning](https://mml-book.github.io/): Ch. 2, 5 and 6 as needed
