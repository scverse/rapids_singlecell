### x.x.x {small}`the-future`

```{rubric} Features
```
* ``tl.leiden`` and ``tl.louvain`` now record the final modularity value in ``adata.uns[key_added]["modularity"]`` 
(scalar for a single resolution, list for multiple resolutions) {pr}`648` {smaller}`J Pintar`

```{rubric} Performance
```


```{rubric} Bug fixes
```


```{rubric} Misc
```
* ``adata.uns[key_added]["params"]["resolution"]`` is now stored as a scalar ``float`` when a single resolution 
is passed to ``tl.leiden`` and ``tl.louvain`` to match behaviour in Scanpy, and as a ``list`` when multiple 
resolutions are passed. Previously it was always stored as a list. {pr}`648`. {smaller}`J Pintar`

```{rubric} Removals
```
