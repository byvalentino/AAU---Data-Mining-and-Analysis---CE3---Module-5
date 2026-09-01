# Module 5 — Reading, and the examination

The twelve oral examination questions for this module are in `../EXAM.md`,
each with what a good answer contains, the commonest wrong answer and why it is
wrong, and one follow-up. Every number quoted there names the key it came from in
`slides/measured.json`. The single written question at the end of this file is the
one published to students in advance; the twelve are for the room.

## Required

**Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014).
*A Survey on Concept Drift Adaptation*. ACM Computing Surveys 46(4), Article 44.**
<https://doi.org/10.1145/2523813> — through the AAU library.

The vocabulary of this module, set out properly: covariate shift against concept
drift, sudden against gradual, detection against adaptation. Read sections 2.1
and 2.2 for the definitions and section 3.2 for the detectors — the earlier
printing of this list sent you to section 4, which is about adaptation rather
than detection. You built the simplest detector today; the survey shows you what
the rest of the family looks like and where the simple one runs out.

**Breck, E., Cai, S., Nielsen, E., Salib, M. & Sculley, D. (2017). *The ML Test
Score: A Rubric for ML Production Readiness and Technical Debt Reduction*. IEEE
International Conference on Big Data.**
<https://research.google/pubs/pub46555/> — free.

Twenty-eight tests, scored. **Seven** of them are the monitoring section — an
earlier printing of this list said four — and you have now implemented three of
those seven. Read it as a checklist to argue with rather than a standard to
adopt: some of the twenty-eight will not apply to a shuttle fleet, and being
able to say which and why is the skill.

## Recommended

**Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. & Lawrence, N. (eds.)
(2009). *Dataset Shift in Machine Learning*. MIT Press.**
<https://mitpress.mit.edu/9780262170055/dataset-shift-in-machine-learning/> —
through the library. Where the distinction the whole module rests on was made
carefully. Chapter 1 is enough.

**Beyer, B., Jones, C., Petoff, J. & Murphy, N. R. (2016). *Site Reliability
Engineering*, chapter 6, "Monitoring Distributed Systems". O'Reilly.**
<https://sre.google/sre-book/monitoring-distributed-systems/> — free to read.

Not about machine learning at all, and the best thing written on the subject of
Lab 3. Their rule — every page must be actionable, and a page that is not is a
bug in the monitor — is older than the field and settles most arguments about
thresholds before they start.

**Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning
Systems*. NeurIPS 28.**
<https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
— free. Nine pages, and the source of "the model is a small box in the middle of
a large diagram". Read it once a year.

**Paleyes, A., Urma, R.-G. & Lawrence, N. D. (2022). *Challenges in Deploying
Machine Learning: A Survey of Case Studies*. Association for Computing Machinery
(ACM) Computing Surveys 55(6), Article 114.**
<https://doi.org/10.1145/3533378> — also free on arXiv, arxiv.org/abs/2011.09926. What actually goes wrong,
collected from published post-mortems. The monitoring section is short, which is
itself informative.

**Huyen, C. (2022). *Designing Machine Learning Systems*, chapters 8 and 9.
O'Reilly.** Through the library. Chapter 8 is the practitioner's version of
blocks two and three, with more attention to what a monitoring stack costs to
run. Chapter 9 is block four: when to retrain, how often, and on which data —
read it beside Lab 4 rather than before it.

**Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly*. NeurIPS 32.**
<https://arxiv.org/abs/1810.11953> — carried over from Module 4, and worth
re-reading now that you have seen a shift that no input-space method can detect.

> Nothing licensed is redistributed in this repository.


**Lieven, C., Beber, M. E., Olivier, B. G., et al. (2020). *MEMOTE for
standardized genome-scale metabolic model testing*. Nature Biotechnology 38(3),
272–276.** <https://doi.org/10.1038/s41587-020-0446-y> — free. Read it beside
Breck et al. A second, independent field built a community test suite that scores
a model automatically — and the transferable point is that its tests are
*structural*, not predictive. A model can score full marks on it and never have
been evaluated out of sample. Your gate needs both arms, and most gates have one.

**Flores-Alsina, X., Rodriguez-Roda, I., Sin, G. & Gernaey, K. V. (2009).
*Uncertainty and sensitivity analysis of control strategies using the benchmark
simulation model No. 1 (BSM1)*. Water Science & Technology 59(3), 491–499.**
<https://doi.org/10.2166/wst.2009.871> — through the AAU library. Optional. An
earlier printing of this list credited this paper to Sin and colleagues; the
first author is Flores-Alsina. Two questions that get confused constantly — which inputs
matter, and how far to trust the answer — asked separately, on a benchmark model
the whole field shares. The shared benchmark is the point: it is what makes two
teams' strategies comparable at all, and it is an artefact, not a method.

**Atabaev, O. & Babaa, M. R. (2026). *Data-efficient hybrid parameter scaling for
accurate microbial bioreactor scale-up*. Bioprocess and Biosystems Engineering
49(5), 1263–1274.** <https://doi.org/10.1007/s00449-026-03314-w> — through the AAU
library. Optional. A model fitted at one scale and then tested at another with
its parameters held fixed. That is the honest form of the question your gate asks
every time it compares a candidate against the model in service.

## The sources behind the definition cards

Each of these is cited on a definition slide, and each is there because the card
grades a formula that came from it rather than from this course's taste.

**Glass, G. V. (1976). *Primary, Secondary, and Meta-Analysis of Research*.
Educational Researcher 5(10), 3–8.**
<https://doi.org/10.3102/0013189X005010003> — through the AAU library. Where the
standardised difference this module measures comes from: a difference of means
divided by a *control group's* standard deviation rather than a pooled one. Read
the three pages on why the denominator is the control's; it is the whole of
block one's argument, written in 1976 about school experiments.

**Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika 41(1/2),
100–115.** <https://doi.org/10.1093/biomet/41.1-2.100> — through the AAU
library. The ancestor of every fixed-reference detector and of block three's run
rule: evidence accumulates against a reference agreed in advance, and a signal
counts only when it persists. Read it for the shape of the argument rather than
for the arithmetic.

**Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. &
Herrera, F. (2012). *A unifying view on dataset shift in classification*.
Pattern Recognition 45(1), 521–530.**
<https://doi.org/10.1016/j.patcog.2011.06.019> — through the AAU library. The
taxonomy on the definition card: covariate shift, concept drift, prior shift,
written once so that two teams can mean the same thing by the same word. Section
2 is enough.

**Lu, J., Liu, A., Dong, F., Gu, F., Gama, J. & Zhang, G. (2019). *Learning
under Concept Drift: A Review*. Institute of Electrical and Electronics
Engineers (IEEE) Transactions on Knowledge and Data Engineering 31(12),
2346–2363.** <https://doi.org/10.1109/TKDE.2018.2876857> — through the AAU
library. Gama and colleagues brought up to date, and the better map of what has
been tried since. Read it after the survey, not instead of it.

**Schenker, N. & Gentleman, J. F. (2001). *On Judging the Significance of
Differences by Examining the Overlap Between Confidence Intervals*. The American
Statistician 55(3), 182–186.**
<https://doi.org/10.1198/000313001317097960> — through the AAU library. Four
pages, and they settle a habit this module also uses: two intervals that do not
overlap are evidence of a difference, but the converse does not hold, so the
rule is conservative rather than correct. Read it before you quote an overlap at
anybody.

**Perdomo, J., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020). *Performative
Prediction*. International Conference on Machine Learning, Proceedings of
Machine Learning Research 119, 7599–7609.**
<https://proceedings.mlr.press/v119/perdomo20a.html> — free. The feedback loop
given a name and a formalism: the distribution a model is judged on is a
function of the model itself, so "the best fit" stops being well defined. Read
the first four pages; the fixed-point machinery afterwards is optional.

**Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A.,
Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F. & Zumar, C. (2018).
*Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Data Engineering
Bulletin 41(4), 39–45.**
<http://sites.computer.org/debull/A18dec/p39.pdf> — free. What a run, a model
signature and a registry alias are, from the people who built the tool block
four uses. Nine pages and no marketing.

**Schelter, S., Biessmann, F., Januschowski, T., Salinas, D., Seufert, S. &
Szarvas, G. (2018). *On Challenges in Machine Learning Model Management*. IEEE
Data Engineering Bulletin 41(4), 5–15.**
<http://sites.computer.org/debull/A18dec/p5.pdf> — free, and in the same issue.
The problem MLflow is an answer to, stated without a product attached: what an
organisation actually has to keep about a model, and why the keeping is the hard
part.

**MLflow documentation — the model registry, and the scikit-learn flavour.**
<https://mlflow.org/docs/latest/ml/model-registry/workflow/> and
<https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html> —
the tool the store in this module is written with. It pins **MLflow 3.15.1**.
This module logs a plain scikit-learn model, so MLflow 3's switch of default
serialisation format to `skops` does not reach it; Module 3, whose pipeline
carries a class written for this course, has to name `cloudpickle` explicitly.
Read `log_model` there if you add a step of your own here, because that is the
moment the default starts to matter.

**Kreuzberger, D., Kühl, N. & Hirschl, S. (2023). *MLOps: Overview, Definition,
and Architecture*. IEEE Access 11, 31866–31879.**
<https://doi.org/10.1109/ACCESS.2023.3262138> — free. The retrain, gate, promote
loop drawn as an architecture, with the roles named. Read it as a vocabulary for
arguing about what your team is missing, not as a design to copy.

**Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A.,
d'Alché-Buc, F., Fox, E. & Larochelle, H. (2021). *Improving Reproducibility in
Machine Learning Research*. Journal of Machine Learning Research 22(164), 1–20.**
<https://jmlr.org/papers/v22/20-303.html> — free. What a record has to contain
before somebody else can repeat what you did. The reproducibility checklist at
the end is the honest version of "every retrain is a run".

**Shankar, S., Garcia, R., Hellerstein, J. M. & Parameswaran, A. G. (2024).
*"We Have No Idea How Models will Behave in Production until Production"*.
Proceedings of the ACM on Human-Computer Interaction 8(CSCW1), Article 206.**
<https://doi.org/10.1145/3653697> — free. Optional, and the most useful optional
item on this list: eighteen interviews with people who deploy models, on what
they actually monitor and why. Read it if you want to know how common the
six-weeks-of-green situation in the exam question really is.

**Neyman, J. (1934). *On the Two Different Aspects of the Representative Method:
The Method of Stratified Sampling and the Method of Purposive Selection*. Journal
of the Royal Statistical Society 97(4), 558–625.**
<https://doi.org/10.2307/2342192> — through the AAU library. Where stratified
sampling is set out, and set against choosing your sample on purpose. Block two's
card cites it because the allocation of a fixed labelling budget between groups
is a decision with a stated price, not a technique. Read the discussion of why
equal allocation is not optimal allocation: the optimal rule weights by each
group's spread and size, which needs numbers nobody has on the first morning, and
this course therefore uses the equal split and says so.

**Simpson, E. H. (1951). *The Interpretation of Interaction in Contingency
Tables*. Journal of the Royal Statistical Society, Series B 13(2), 238–241.**
<https://doi.org/10.1111/j.2517-6161.1951.tb00088.x> — through the AAU library,
and already met in Module 2. Four pages. Block two now measures the accuracy
version of it: the aggregate falls while no group's accuracy falls, because the
mix moved towards the group that was already served worse. If you read one paper
twice in this course, read this one.

**Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly: An Empirical
Study of Methods for Detecting Dataset Shift*. Advances in Neural Information
Processing Systems 32.** <https://arxiv.org/abs/1810.11953> — free. What happens
when a monitor watches many columns instead of one, which is the situation you
will actually be handed. Read it for the aggregation problem: one test per
dimension and a union over them needs the level corrected, or the detector fires
on data that has not moved. Block three measures the same thing without any
testing machinery at all — the quiet floor of a family of columns is the worst
quiet day of the worst column, and here it is nearly three times one column's.

**European Union (2024). *Regulation (EU) 2024/1689, the Artificial Intelligence
Act*, Article 72 and Annex IV.**
<https://eur-lex.europa.eu/eli/reg/2024/1689/oj> — free. Read Article 72
paragraphs 1 to 3 only; it is one page. A provider of a high-risk system must
establish and document a post-market monitoring system that actively and
systematically collects, documents and analyses performance data throughout the
system's lifetime, against a written plan that forms part of the technical
documentation. The point of putting it on the reading list is not the law: it is
that the plan it describes is the thing this module built. The Annex III
obligations apply from 2 December 2027 after Regulation (EU) 2026/1744, the
Digital Omnibus on artificial intelligence. This is teaching material and not
legal advice.

## The written exam question for Module 5, published in advance

> **A monitoring dashboard for a deployed model has shown a flat green line for
> six weeks. The team believes the model is healthy. What would you check before
> agreeing, and what would you add?**

A strong answer starts by asking what the green line is measuring, because the
three levels fail differently. If it is an input-distribution measure against a
*moving* baseline, a flat line is what you would see whether the world was
stable or had changed once and stayed changed — this module measured that case:
fourteen genuinely shifted days, one alert. If the reference is fixed, a flat
line does mean the inputs have not moved, which still says nothing about whether
the relationship between inputs and target has changed. The measured example
here is a week in which the input shift stayed at 0.60, the model's output rate
stayed at 0.68, and accuracy fell by ten points.

So the answer asks: is there any label-based measure at all, even a small
sampled one? If not, the dashboard cannot detect concept drift by construction,
and the first thing to add is a bought-truth sample with an interval round it —
two hundred rows a day is a morning's work and gives roughly ±5 percentage
points, which is enough to see a ten-point fall within a week. It also asks what
the alerting threshold is and where it came from: a threshold below the measured
quiet floor produces noise that gets the monitor ignored, and one far above the
achievable signal produces exactly the flat green line under discussion.

Finally it asks what is *not* being watched, and that has two halves. In this
course's archive the variable that explained the change — who was driving — was
not a model input and was therefore on nobody's dashboard. A monitor that watches
only the columns the model consumes cannot see the reason the model is wrong.

And a green line is an *average*. Split the same labelling budget by the groups
the system serves and this module's own stable week — before anything changed at
all — reads 0.910 for one group and 0.646 for the other. An aggregate that has
never moved is consistent with a model that has always served somebody badly, and
with a model whose service is collapsing for one group while the mix quietly
moves the other way. The question "flat compared with what, and averaged over
whom?" is the whole answer in eight words.
