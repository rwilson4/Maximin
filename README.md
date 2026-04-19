# maximin

Robust optimization solver for problems of the form:

```
maximize_c  minimize_{beta}  g(c; beta)
subject to  beta in S
            c in C
```

where `c` is a decision variable, `beta` is an uncertain parameter known only
to lie within some uncertainty set `S`, and `C` is the feasible set for
decisions.
