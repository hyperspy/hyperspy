### Requirements
* Read the [developer guide](https://hyperspy.org/hyperspy-doc/current/dev_guide/index.html).
* Base your pull request on the [correct branch](https://hyperspy.org/hyperspy-doc/current/dev_guide/git.html#semantic-versioning-and-hyperspy-main-branches).
* Filling out the template; it helps the review process and it is useful to summarise the PR.
* This template can be updated during the progression of the PR to summarise its status. 

*You can delete this section after you read it.*

### Description of the change
A few sentences and/or a bulleted list to describe and motivate the change:
- Change A.
- Change B.
- etc.

### Progress of the PR
- [ ] Change implemented (can be split into several points),
- [ ] update docstring (if appropriate),
- [ ] update user guide (if appropriate),
- [ ] add entry to `CHANGES.rst` (if appropriate),
- [ ] add tests,
- [ ] ready for review.

### Minimal example of the bug fix or the new feature
```python
import hyperspy.api as hs
import numpy as np
s = hs.signals.Signal1D(np.arange(10))
# Your new feature...
```
Note that this example can be useful to update the user guide.

