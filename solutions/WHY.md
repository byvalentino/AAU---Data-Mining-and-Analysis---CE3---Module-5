# Why these solutions look like this

## Lab 1 — the yardstick has to hold still

Every drift measure is a difference divided by a spread, and the argument is
always about which spread. Divide by the reference period's and you have a
yardstick: 0.60 today means the same thing it meant last week. Divide by today's
and the yardstick stretches to fit whatever you measure with it.

On this stream that is not a rounding difference. The days that moved 0.60
reference standard deviations read 0.40 against their own spread — because the
shift arrives through a group of rows that are both faster *and* more variable,
so the denominator grows with the numerator. A threshold of 0.5 catches the
first number and misses the second entirely. The monitor stays silent for
fourteen days.

The moving baseline fails in a more interesting way, because it never looks
wrong. It fires once, on the day of the step, which is exactly what you want. It
is then silent for thirteen days during which the model is losing accuracy, and
the silence is not a bug: each shifted day really is much like the day before.
A monitor that only ever asks "is today like yesterday?" cannot see a world that
changed once and stayed changed. Written down like that it is obvious. Sitting
in an on-call rota looking at a flat green line, it is not.

The fixed reference has the opposite failure, and it is the subject of Lab 3: it
keeps shouting about a change everybody already knows about. That is a nuisance
you can engineer away. Blindness is not.

## Lab 2 — three levels, and the one people skip

The middle level is free and almost nobody watches it. You do not need a single
label to notice that the share of readings called `aboard` went from 55 to 69 per
cent. The model is telling you that its inputs moved into a region where it
behaves differently, in its own words, at no cost, on the same day.

The measured table is the module:

| week | inputs | output | accuracy | 95 per cent interval | labels |
|---|---|---|---|---|---|
| stable, days 7–13 | 0.008 | 0.551 | 0.891 | [0.873, 0.906] | 1,400 |
| covariate shift, days 14–20 | 0.615 | 0.687 | 0.816 | [0.795, 0.836] | 1,400 |
| concept drift, days 21–27 | 0.603 | 0.683 | 0.711 | [0.687, 0.735] | 1,400 |

Read the last two rows across. The inputs are in the same place. The model is
saying the same thing. Accuracy has fallen by a further ten points, and the
intervals do not overlap, so that fall is established rather than suspected.
Both free levels are blind to it, and no amount of care in choosing a
distribution measure would help: there is nothing in the inputs to find.

The interval is not decoration. One day of bought truth — 200 hand-checks — gives
day 19 as [0.712, 0.827] and day 24 as [0.623, 0.750]. Those overlap. On a single
day's labels you can say the model got worse; you cannot say which of the two
things happened. A week's labels separate them. That is what Module 4's
arithmetic was for: halving an interval costs four times the labels, so the
budget decides what questions you are allowed to ask.

`buy_truth` samples without replacement and resets the index first. Both are
small points that produce silently wrong numbers: sampling with replacement
hand-checks the same row twice and narrows the interval you were relying on, and
positional picks against a pooled frame with duplicated indices line up
predictions with the wrong labels.

### Buying truth by segment, and what the allocation costs

An accuracy for a served week is an average over everybody the system served, and
an average hides whoever it is averaged with. `crew` is not one of the model's
inputs, so no number about it had appeared anywhere in this course until
`segment_accuracy` was written.

Spend the same 1,400 labels, split evenly:

| week | no driver | a person driving | share with a driver |
|---|---|---|---|
| stable, 7–13 | 0.910 [0.887, 0.929] | 0.646 [0.610, 0.680] | 0.086 |
| covariate, 14–20 | 0.914 | 0.639 | 0.404 |
| concept, 21–27 | 0.909 | 0.389 | 0.415 |

Two things fall out of that table and neither is on a slide before it is
measured.

The model has served one group far worse **since the day it was switched on**.
The two intervals in the stable week are nowhere near each other, and nothing in
this module before Lab 2's segment table would ever have said so. Until this
pass the deck asserted that a fairness failure "fails in exactly the silent way
block two describes" and the code could not show it, because the truth sampler
was uniform. A claim the code contradicts is exactly what this course spends five
modules arguing against, so the sampler was fixed rather than the sentence.

And the first weekly drop is not a drop in accuracy for anybody.
`which_segments_moved` returns an empty list between the stable and the covariate
week: the aggregate fell 0.891 to 0.816 and both groups sat still. What moved is
the mix — 0.086 to 0.404 of rows with somebody driving — towards the group the
model already served worse. That is Simpson's paradox in its accuracy form
(Simpson, 1951), and it is the difference between "the model degraded, schedule a
retrain" and "the population we serve changed, and we were always worse at this
part of it". The second drop, 0.816 to 0.711, really is a loss and it falls on
one group only. Two identical-looking falls, two opposite events.

Then the allocation, which is a decision with a stated price rather than a
technique (Neyman, 1934). At the same 1,400 labels:

| | small group | large group |
|---|---|---|
| uniform | 125 rows, interval 0.1581 wide | 1,275 rows, 0.0316 |
| stratified | 700 rows, 0.0707 | 700 rows, 0.0425 |

Stratifying is not a request for more money; it moves the money, and the price is
the large group's precision. Equal allocation is not optimal allocation — the
optimal rule weights by each group's spread and size, which needs numbers nobody
has on the first morning — so the lab uses the equal split and says so.

And width is not only precision. On this budget and this seed the uniform draw
reports the small group as having moved between the stable and the covariate
week, which the stratified draw refuses and which every row of the data refuses
too. A 125-row interval is wide enough for that to happen often, so the failure
is a property of the width rather than of the draw.

## Lab 3 — a correct alert and a useful alert are different things

Fourteen pages, every one of them true, about one change. After the third
nobody reads them, and the monitor has been switched off without anyone
deciding to switch it off. That is the ordinary way monitoring dies.

Three decisions turn fourteen into three:

- **the threshold**, which is 0.5 here because the measured quiet floor is
  0.0573 — the largest the shift statistic gets over days 1 to 13, when nothing
  at all is happening. Sampling alone moves it that far. The lowest candidate
  that clears the floor is 0.10, so 0.5 has a factor of nine in hand. Quote the
  floor whenever you quote the threshold; a threshold without one is taste.
- **confirmation**, two days in a row, which costs exactly one day of delay.
  Measured, not assumed.
- **the cooldown**, five days, which is the difference between "this is
  happening" and "this is still happening".

The cooldown cannot delay the first page, because there is nothing before it to
measure from. Only confirmation costs time at the start, and one day is usually
worth it — a single day over the line is often sampling, and waking somebody for
sampling is how you teach them to ignore the pager.

Module 4 makes the darker case worth remembering: there the noise floor at ten
bins was 0.28, above banking's conventional 0.25 threshold, and no safe
threshold existed at all. `lowest_safe_threshold` returns `None` for that
situation on purpose. The honest answer to "where should the line go?" is
sometimes "nowhere, until you fix the measure."

### Four columns, and then forty

Everything above watches one column, which is enough to show what a yardstick is
and is not what anybody does in production. `shift_by_column` points the same
measure at all four of the model's inputs and `family_quiet_floor` takes the
maximum over columns as well as over days, because the alert fires when any
column fires.

| column | quiet floor, days 1–13 |
|---|---|
| speed | 0.0573 |
| rssi1 | 0.1507 |
| rssi2 | 0.0968 |
| rssiC | 0.0892 |

The noisiest quiet column is a beacon, not the column the threshold was chosen
against. The floor to clear is 0.1507 rather than 0.0573, and the lowest safe
candidate moves from 0.10 to 0.25. Nothing about the measure changed; the number
of chances a quiet day gets to be unlucky did. Held at the one-column line, 1 of
the 52 quiet column-days fires — which sounds small until it is forty columns and
a year.

`expected_false_alarms` is one multiplication and it is the point of the section:
40 columns each allowed to fire on a twentieth of quiet days is 2.0 alarms a day
and 60.0 a month, from nothing happening at all (Rabanser, Günnemann & Lipton,
2019, make the same point from the testing side). Two defences and they are the
same defence: divide the per-column level by the number of columns, or measure
the family's floor and put the line above that. This module already knew how to
do the second.

The missing readings are dropped rather than filled. An absent beacon is not a
low reading, and filling it with the training-time sentinel would make a change
in *how often* a beacon is heard read as a change in *how strong* it is — which
is a real monitor, and a different one, and Module 2 built it.

### The alert verdict: page, ticket or nothing

The slide "What an alert should say" was the best ninety seconds in the module
and was graded nowhere. `alert_verdict` is that slide as a function, and its
three questions are asked in an order that is itself the content:

1. is the movement inside the quiet floor? Then `nothing`. At or under the floor
   is the size of the largest quiet day, and no threshold that clears the floor
   could have fired on it — which is why Lab 3's own rule makes a safe candidate
   *strictly* above the floor.
2. is it at the outcome level with bought truth in hand? Then `page`. That is the
   only level that says somebody was served worse, and bought truth is what
   establishes it rather than suggests it.
3. otherwise `ticket`. Nobody can act at three in the morning on a number that
   may still be noise, and what an input or output movement justifies is a
   morning of buying truth.

`nothing` is on the list because a monitor that only ever finds drift is a
monitor that will be ignored, and because Module 4's verdict on the real archive
was exactly that call.

The reason is graded as hard as the call, and the check's fixtures are built so
that a lookup cannot survive: the same numbers appear with different levels and
different truth flags, and four of the cases change one quantity and expect a
different call. Three cheating implementations are recorded in `DONE.md`,
including one that gets every call right and copies its reason off the slide —
it fails because the slide's 0.76 metres per second is not a number in that
evidence.

## Lab 4 — the reflex, the repair, and the pointer

Measured on days 25 to 27, which nothing was trained on:

| | accuracy |
|---|---|
| the model in service | 0.688 |
| retrained on fresh rows, same four inputs | 0.883 |
| retrained, plus the explaining variable | 0.909 |
| what it managed before anything changed | 0.891 |

Retraining on fresh rows is the reflex and it very nearly works: the model
learns the new relationship from new examples and recovers most of the loss.
Adding `crew` is the repair, and it ends up slightly better than the model ever
was, because the variable was informative before the drift too — it simply never
mattered enough to notice.

That variable is the archive's `mode` column under another name. In the two real
days behind this course, 9.1 per cent of readings on 22 January were manually
driven against 41.0 per cent on 23 January, and nothing in the pipeline was
watching it. The retraining loop closes; the lesson is that the loop would not
have needed closing if somebody had been monitoring the column that explains the
model rather than only the columns the model consumes.

Two mechanical points, both from Module 3 and both unchanged:

**Gate on days neither model was trained on.** A candidate scored on its own
training days always wins. The check enforces this by measuring the fresh-rows
candidate on days 21 to 24 and getting 0.896 instead of 0.883 — a small
difference here, and a large one whenever a model overfits.

**Release is moving a pointer.** `promote` writes a new artefact and changes
`approved`; it never overwrites the old file. That is what makes `rollback` a
one-line operation rather than a retraining job, and it is why the history is
append-only: rolling back does not un-release anything, it records that you
released and thought better of it.

### The registry, and what the signature refuses

Lab 4 writes two registries, and the check compares them. The fifty-line one is
Module 3's: `registry.json` names an approved version and one pickle sits beside
it per version. The platform one is a local MLflow store — `mlruns.db` for the
runs and the registered versions, `mlartifacts/` for the models — and it is the
same three ideas under different names.

**Every retrain is a run.** `retrain()` trains and then calls
`service.models.log_run`, which records the training window read off the frame
(first day, last day), the feature list, the seed and the row count as
parameters; the accuracy on the training days *and* on the gate days as metrics;
and the model itself with its signature, an input example and the package
versions actually installed. Written once and never edited. The check counts
runs before and after, so a retrain that forgets to log is caught, and it reads
the run back **by content** — never by identifier, because identifiers are
random and two stores holding the same work have different ones.

**Release is moving an alias.** `promote()` moves `champion` to the version the
run registered, and records that version in the pickle registry's
`mlflow_version` map so the two stores can be compared in one line. `rollback()`
moves both back. A release recorded in one store and not the other leaves them
disagreeing about what is answering requests, which is worse than either being
wrong alone — so the check asserts they agree after both promotions and after
the rollback.

**The signature is the guarantee, and it is run rather than asserted.** The
model is logged with a signature: the column names, their types and their order,
inferred at logging time and enforced at every prediction. `ask_champion()` sends
three requests and reports what came back:

| request | what happens |
|---|---|
| the four columns, in order | answered |
| the same rows, columns reversed | answered, and the answers are identical |
| `rssi1` renamed to `rssi_one` | refused, before a single row is scored |
| `rssi2` missing | refused |

That is the answer to "how do I guarantee the columns are not switched". The
guarantee lives in the artefact rather than in the discipline of whoever calls
it. Serve the raw estimator instead and the reordered frame is either answered
with different numbers or raises an error nobody wrote a message for; the check
has a cheat for exactly that and it exits 1.

The margin, reconciled with Module 3. Module 3's gate opened when a candidate
was not worse; Module 5's asks for 0.01 more accuracy. Both are choices, and the
principled version is neither: it is the Wilson half-width on however many
labels the gate days hold, so that "better" means "better by more than the
measurement's own uncertainty". The number 0.01 is on a slide, in the file the
student opens, and in the check, because a margin nobody can see is a margin
nobody can argue with.

The margin does real work. At one point, both candidates clear the bar and the
loop finishes on v3. At five points, the first candidate is released — it is
nineteen points better — and the second is refused, because it beats what is
*now in service* by 2.6 points rather than beating the retired champion by
twenty-two. Gating each candidate against the model actually running is the
difference between a champion–challenger loop and a queue of releases justified
by comparisons nobody is making any more.

### The release verdict, and the bar under the bar

`passes_gate` answers one arithmetic question. `release_verdict` answers the
larger one, and the parts it adds are the parts the gate cannot see.

The Wilson half-width is one of them, and it turns the slide's sentence into
something gradeable. On every one of the 3,600 gate-day rows it is 0.0105; on the
600 rows a team buying 200 labels a day for three days would actually have, it is
0.0257. So the same gain of 0.0132 promotes on one label count and is held on the
other — same two models, same two accuracies, same stated bar, different call.
A gain smaller than the half-width is a gain the measurement cannot tell from
nought, whatever the stated margin says, which is why the margin is a floor under
the half-width rather than a substitute for it.

Admissibility is another. A candidate scored on its own training days scores
0.8962 against 0.8831 on days it never saw, and that number cannot justify
anything — the right call is `hold` and measure again, not `promote` and not
`refuse`.

And a released regression is asked about first, before any candidate. Rolling
back is a pointer move that takes seconds, and production is losing accuracy
while a release is being argued about.

### What check 4 stopped grading

An earlier version of check 4 graded run counts, MLflow filter strings and alias
look-ups in lockstep across the two registries — twelve assertions, most of which
examined whether a student could drive a tool. Seven of them are gone: the
metric-key membership test that the next line re-asserted by value, the seed
parameter, the second run count, the second filter string, the in-memory
`approved` field that the on-disk read re-checks, and both `mlflow_version`
lockstep maps. What is left still proves the two stores agree and that a run
records the window it was trained on, because those are the module's actual
claims. Twelve release decisions took the place of the rest. Driving the tool is
worth one mark; knowing whether to release is the block.
