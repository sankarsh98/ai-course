# Bibliography

Every entry below is the "Further reading" list of one module, collected in one place. Each source was looked up and checked while the module was written; see the verification notes for how, and for what is weaker. Inline citations in the module text that are not in these lists are not repeated here.

## Level 0: Foundations

### L0-01: Python and NumPy for ML

- **Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, et al., "Array programming with NumPy"** (2020, Nature 585, 357 to 362; arXiv:2006.10256). https://www.nature.com/articles/s41586-020-2649-2
  *Read for:* the design of the ndarray (strides, views, broadcasting, vectorization) from NumPy's maintainers. Open access. *Level:* intro.
- **Jake VanderPlas, "Python Data Science Handbook"** (2016, O'Reilly; free online edition). https://jakevdp.github.io/PythonDataScienceHandbook/
  *Read for:* chapter 2, a gentle tour of arrays, ufuncs, aggregations, broadcasting, masks and fancy indexing. *Level:* intro.
- **NumPy Developers, "Broadcasting"** (NumPy v2.5 user guide, accessed October 2026). https://numpy.org/doc/stable/user/basics.broadcasting.html
  *Read for:* the authoritative broadcasting rules with examples of compatible and incompatible shapes. *Level:* intro.
- **NumPy Developers, "Indexing on ndarrays"** (NumPy v2.5 user guide, accessed October 2026). https://numpy.org/doc/stable/user/basics.indexing.html
  *Read for:* exactly when indexing returns a view or a copy, and repeated-index assignment. *Level:* intermediate.
- **NumPy Developers, "Random Generator"** (NumPy v2.5 reference, accessed October 2026). https://numpy.org/doc/stable/reference/random/generator.html
  *Read for:* `default_rng`, PCG64, and the lack of a cross-version stream guarantee. *Level:* intro.
- **David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"** (1991, ACM Computing Surveys 23(1)). https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
  *Read for:* the classic treatment of rounding error, cancellation, IEEE 754 special values and subnormals. Free Oracle reprint. *Level:* intermediate to advanced.
- **Python Software Foundation, "Floating-Point Arithmetic: Issues and Limitations"** (Python tutorial, accessed October 2026). https://docs.python.org/3/tutorial/floatingpoint.html
  *Read for:* a short practical explanation of why 0.1 is inexact, plus `math.isclose` and `math.fsum`. *Level:* intro.
- **Pierre Blanchard, Desmond J. Higham, Nicholas J. Higham, "Accurately computing the log-sum-exp and softmax functions"** (2021, IMA Journal of Numerical Analysis 41(4), 2311 to 2330; preprint arXiv:1909.03469). https://arxiv.org/abs/1909.03469
  *Read for:* rounding-error analysis of the shifted log-sum-exp and softmax. Journal version may be paywalled; the preprint is free. *Level:* advanced.
- **Python Software Foundation, "timeit" and "The Python Profilers"** (Python library documentation, accessed October 2026). https://docs.python.org/3/library/timeit.html and https://docs.python.org/3.12/library/profile.html
  *Read for:* timing snippets correctly (and why to use the minimum), and running and reading `cProfile`. *Level:* intro.
- **Matplotlib Developers, "Quick start guide"** (Matplotlib 3.11 documentation, accessed October 2026). https://matplotlib.org/stable/users/explain/quick_start.html
  *Read for:* Figures, Axes and the explicit versus implicit interfaces. *Level:* intro.

### L0-02: Linear Algebra for ML

- **Gilbert Strang, "18.06 Linear Algebra"** (Spring 2010, MIT OpenCourseWare). https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
  *Read for:* complete video lectures, problem sets and exams for a classic first course, covering this module in more depth. Free, CC BY-NC-SA. *Level:* intro.
- **Grant Sanderson (3Blue1Brown), "Essence of linear algebra"** (2016, video series). https://www.3blue1brown.com/lessons/eola-preview
  *Read for:* animated intuition for linear maps, determinants and eigenvectors; watch before or alongside this module. *Level:* intro.
- **Stephen Boyd and Lieven Vandenberghe, "Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares"** (2018, Cambridge University Press). https://web.stanford.edu/~boyd/vmls/
  *Read for:* an applied, data-oriented treatment of vectors, matrices and least squares that assumes little background. Free PDF. *Level:* intro.
- **Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, "Mathematics for Machine Learning"** (2020, Cambridge University Press). https://mml-book.github.io/
  *Read for:* chapters 2 to 4 (linear algebra, analytic geometry, matrix decompositions), written to prepare for ML. Free PDF. *Level:* intro to intermediate.
- **NumPy Developers, "Linear algebra (numpy.linalg)"** (NumPy v2.5 reference, accessed October 2026). https://numpy.org/doc/stable/reference/routines.linalg.html
  *Read for:* the functions used here, the `@` operator, and the note that NumPy relies on BLAS and LAPACK, whose threading may need control. The linked `eigh` page documents real eigenvalues in ascending order. *Level:* intro.
- **Carl Eckart and Gale Young, "The approximation of one matrix by another of lower rank"** (1936, Psychometrika 1(3), 211 to 218). https://doi.org/10.1007/BF02288367
  *Read for:* the original optimality result for truncated SVD. Publisher version may be paywalled. *Level:* advanced.
- **Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, "LoRA: Low-Rank Adaptation of Large Language Models"** (2021, arXiv:2106.09685). https://arxiv.org/abs/2106.09685
  *Read for:* low-rank factorized weight updates for cheap fine-tuning, a direct application of section 6. *Level:* intermediate.
- **Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig, "Linguistic Regularities in Continuous Space Word Representations"** (2013, NAACL-HLT, 746 to 751). https://aclanthology.org/N13-1090/
  *Read for:* the vector-offset view of analogies and how it was evaluated. Open access. *Level:* intermediate.
- **Kawin Ethayarajh, "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"** (2019, EMNLP; arXiv:1909.00512). https://arxiv.org/abs/1909.00512
  *Read for:* cosine-based evidence that contextual embeddings are anisotropic in every layer. *Level:* intermediate to advanced.
- **Harald Steck, Chaitanya Ekanadham, Nathan Kallus, "Is Cosine-Similarity of Embeddings Really About Similarity?"** (2024, ACM Web Conference 2024 Companion; arXiv:2403.05440). https://arxiv.org/abs/2403.05440
  *Read for:* why cosine similarities of learned embeddings can be arbitrary, including the rescaling argument in script 06. *Level:* intermediate.

### L0-03: Calculus and Automatic Differentiation

- **Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, "Automatic Differentiation in Machine Learning: a Survey"** (2018, Journal of Machine Learning Research 18(153); arXiv:1502.05767). <https://arxiv.org/abs/1502.05767>
  *Read for:* forward and reverse mode, the evaluation-trace notation used here, cost results and history, all in one free survey. *Level:* intermediate.
- **Andrej Karpathy, "micrograd"** (GitHub repository, MIT license). <https://github.com/karpathy/micrograd>
  *Read for:* a compact scalar autograd engine with a small neural network library, tested against PyTorch; compare its design with `engine.py`. *Level:* intro.
- **Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"** (2022, lecture 1 of the Neural Networks: Zero to Hero course). <https://karpathy.ai/zero-to-hero.html>
  *Read for:* about two and a half hours building such an engine live, assuming basic Python and high-school calculus. *Level:* intro.
- **Grant Sanderson (3Blue1Brown), "Visualizing the chain rule and product rule"** (Essence of Calculus, chapter 5). <https://3blue1brown.com/lessons/chain-rule-and-product-rule>
  *Read for:* a visual intuition for why rates multiply. *Level:* intro.
- **Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, "Mathematics for Machine Learning"** (2020, Cambridge University Press; free PDF from the authors). <https://mml-book.github.io/>
  *Read for:* chapter 5, Vector Calculus, including gradients of vector- and matrix-valued functions and section 5.6 on backpropagation and automatic differentiation. *Level:* intro to intermediate.
- **Stanford CS231n, "Backpropagation, Intuitions"** (course notes). <https://cs231n.github.io/optimization-2/>
  *Read for:* backprop as local gates in a circuit, gradient accumulation at forks, and vectorized gradients. *Level:* intro.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML version). <https://www.deeplearningbook.org/>
  *Read for:* section 6.5, Back-Propagation and Other Differentiation Algorithms. *Level:* intermediate.
- **JAX authors, "The Autodiff Cookbook"** (JAX documentation). <https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>
  *Read for:* JVPs, VJPs, `jacfwd` versus `jacrev`, and Hessian-vector products in a system that exposes both modes. *Level:* intermediate.
- **PyTorch, "Autograd mechanics"** and **"torch.Tensor.backward"** (PyTorch documentation). <https://docs.pytorch.org/docs/stable/notes/autograd.html> and <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html>
  *Read for:* how a production define-by-run engine records graphs and saves tensors, and the contract that gradients accumulate in leaves. *Level:* intermediate.
- **David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, "Learning representations by back-propagating errors"** (1986, Nature 323, 533 to 536; paywalled at the publisher). <https://doi.org/10.1038/323533a0>
  *Read for:* the short paper that popularized backpropagation for networks with hidden units. *Level:* intermediate.

### L0-04: Probability and Statistics

- **Joseph K. Blitzstein, Jessica Hwang, "Introduction to Probability"** (2nd edition, 2019, CRC Press; free online version from the authors, linked from Harvard's Statistics 110 site). <https://probabilitybook.net/>
  *Read for:* an example-driven first course in probability, conditioning and the common distributions. *Level:* intro.
- **Daniel Kunin, Jingru Guo, Tyler Dae Devlin, Daniel Xiang, "Seeing Theory"** (Brown University, interactive website, archived). <https://seeing-theory.brown.edu/>
  *Read for:* interactive visualizations of distributions and of frequentist and Bayesian inference. *Level:* intro.
- **Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, "Mathematics for Machine Learning"** (2020, Cambridge University Press; free PDF). <https://mml-book.github.io/>
  *Read for:* chapter 6, Probability and Distributions, including Bayes' theorem, the Gaussian and conjugacy. *Level:* intro to intermediate.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML). <https://www.deeplearningbook.org/>
  *Read for:* chapter 3, Probability and Information Theory, a compact summary aimed at deep learning. *Level:* intro to intermediate.
- **Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction"** (2022, MIT Press; free draft PDF under CC BY-NC-ND). <https://probml.github.io/pml-book/book1.html>
  *Read for:* the probabilistic view of ML end to end: MLE, MAP, Bayesian inference, information theory. *Level:* intermediate.
- **David J. C. MacKay, "Information Theory, Inference, and Learning Algorithms"** (2003, Cambridge University Press; free to view on screen from the author's site). <https://www.inference.org.uk/itila/book.html>
  *Read for:* entropy, KL and Monte Carlo methods in unusual depth. *Level:* intermediate to advanced.
- **Claude E. Shannon, "A Mathematical Theory of Communication"** (1948, Bell System Technical Journal 27, 379 to 423 and 623 to 656; publisher copy may be paywalled). <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>
  *Read for:* the founding paper of information theory, where entropy is defined. *Level:* advanced.
- **Bradley Efron, "Bootstrap Methods: Another Look at the Jackknife"** (1979, The Annals of Statistics 7(1), 1 to 26; open access). <https://projecteuclid.org/euclid.aos/1176344552>
  *Read for:* the original bootstrap paper, built around worked examples. *Level:* advanced.
- **Lawrence D. Brown, T. Tony Cai, Anirban DasGupta, "Interval Estimation for a Binomial Proportion"** (2001, Statistical Science 16(2), 101 to 133). <https://doi.org/10.1214/ss/1009213286>
  *Read for:* why the Wald interval is unreliable and which intervals to use instead. *Level:* intermediate.
- **Ronald L. Wasserstein, Nicole A. Lazar, "The ASA's Statement on p-Values: Context, Process, and Purpose"** (2016, The American Statistician 70(2), 129 to 133). <https://doi.org/10.1080/00031305.2016.1154108>
  *Read for:* six short principles on what p-values do and do not mean. *Level:* intro.
- **Thomas G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"** (1998, Neural Computation 10, 1895 to 1923). <https://doi.org/10.1162/089976698300017197>
  *Read for:* which tests to trust when comparing learning algorithms, including McNemar's test and the 5x2 cv test. *Level:* intermediate.
- **SciPy developers, "Statistical functions (scipy.stats)"** (SciPy documentation, v1.18.0 as of October 2026). <https://docs.scipy.org/doc/scipy/reference/stats.html>
  *Read for:* the distributions, hypothesis tests and `bootstrap` function you will use in practice. *Level:* intro.

### L0-05: Optimization

- **Gabriel Goh, "Why Momentum Really Works"** (2017, Distill). https://distill.pub/2017/momentum/
  *Read for:* an interactive explanation of gradient descent and momentum on quadratics. *Level:* intro.
- **Sebastian Ruder, "An overview of gradient descent optimization algorithms"** (2016, arXiv:1609.04747). https://arxiv.org/abs/1609.04747
  *Read for:* a compact tour of momentum, Nesterov, Adagrad, RMSProp, Adam and later variants in one notation. *Level:* intro.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning", Chapter 8** (2016, MIT Press; free online). https://www.deeplearningbook.org/
  *Read for:* the textbook view of SGD, ill-conditioning, saddles, adaptive and second-order methods. *Level:* intermediate.
- **Stephen Boyd, Lieven Vandenberghe, "Convex Optimization"** (2004, Cambridge University Press; free PDF from the authors). https://web.stanford.edu/~boyd/cvxbook/
  *Read for:* convexity, gradient descent, backtracking line search and Newton's method with proofs (Chapters 2, 3 and 9). *Level:* intermediate.
- **Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic Optimization"** (2015, ICLR; arXiv:1412.6980). https://arxiv.org/abs/1412.6980
  *Read for:* the original algorithm, its defaults and the bias-correction argument. *Level:* intermediate.
- **Ilya Loshchilov, Frank Hutter, "Decoupled Weight Decay Regularization"** (2019, ICLR; arXiv:1711.05101). https://arxiv.org/abs/1711.05101
  *Read for:* why L2 and weight decay differ under Adam, and AdamW. *Level:* intermediate.
- **Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, Benjamin Recht, "The Marginal Value of Adaptive Gradient Methods in Machine Learning"** (2017, NeurIPS; arXiv:1705.08292). https://arxiv.org/abs/1705.08292
  *Read for:* the case that adaptive methods can generalize worse than SGD. *Level:* intermediate.
- **Yann Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"** (2014, arXiv:1406.2572). https://arxiv.org/abs/1406.2572
  *Read for:* the argument that saddles, not local minima, dominate high-dimensional losses. *Level:* advanced.
- **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "Understanding deep learning requires rethinking generalization"** (2017, ICLR; arXiv:1611.03530). https://arxiv.org/abs/1611.03530
  *Read for:* the random-label experiments that reframe what optimization success means. *Level:* intermediate.
- **Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"** (2017, ICLR; arXiv:1609.04836). https://arxiv.org/abs/1609.04836
  *Read for:* the evidence linking large batches, sharp minima and worse generalization. *Level:* advanced.
- **Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio, "Sharp Minima Can Generalize For Deep Nets"** (2017, ICML, PMLR 70; arXiv:1703.04933). https://arxiv.org/abs/1703.04933
  *Read for:* why sharpness changes under reparametrization and cannot explain generalization alone. *Level:* advanced.
- **Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar, "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"** (2021, ICLR; arXiv:2103.00065). https://arxiv.org/abs/2103.00065
  *Read for:* evidence that real training runs at sharpness near $2/\eta$. *Level:* advanced.

## Level 1: Classical machine learning

### L1-01: The ML workflow and evaluation

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning, with Applications in Python"** (2023, Springer; free PDF). https://www.statlearning.com/
  *Read for:* the gentlest full treatment of resampling, CV and bias-variance, with Python labs. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani, Jerome Friedman, "The Elements of Statistical Learning", 2nd ed., Chapter 7** (2009, Springer; free PDF). https://hastie.su.domains/ElemStatLearn/
  *Read for:* the bias-variance decomposition, right and wrong CV (7.10.2), and conditional versus expected error (7.12). *Level:* intermediate.
- **scikit-learn developers, "Common pitfalls and recommended practices"** (scikit-learn 1.9.1 User Guide, checked October 2026). https://scikit-learn.org/stable/common_pitfalls.html
  *Read for:* runnable examples of preprocessing leakage and pipelines. *Level:* intro.
- **scikit-learn developers, "Probability calibration"** (scikit-learn 1.9.1 User Guide, checked October 2026). https://scikit-learn.org/stable/modules/calibration.html
  *Read for:* reliability diagrams and sigmoid versus isotonic calibration. *Level:* intro.
- **Shachar Kaufman, Saharon Rosset, Claudia Perlich, Ori Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance"** (2012, ACM Transactions on Knowledge Discovery from Data 6(4); publisher version may be paywalled). https://doi.org/10.1145/2382577.2382579
  *Read for:* a formal definition of leakage and the learn-predict separation principle. *Level:* intermediate.
- **Sayash Kapoor, Arvind Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science"** (2023, Patterns 4(9):100804; open access; arXiv:2207.07048). https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/
  *Read for:* the eight-type leakage taxonomy, its prevalence, and model info sheets. *Level:* intro.
- **Sudhir Varma, Richard Simon, "Bias in error estimation when using cross-validation for model selection"** (2006, BMC Bioinformatics 7:91; open access). https://doi.org/10.1186/1471-2105-7-91
  *Read for:* why tuning by CV makes the CV score optimistic, and nested CV. *Level:* intermediate.
- **Stephen Bates, Trevor Hastie, Robert Tibshirani, "Cross-Validation: What Does It Estimate and How Well Does It Do It?"** (2024, Journal of the American Statistical Association 119(546); arXiv:2104.00673). https://arxiv.org/abs/2104.00673
  *Read for:* what CV estimates and why naive CV intervals undercover. *Level:* advanced.
- **Tom Fawcett, "An introduction to ROC analysis"** (2006, Pattern Recognition Letters 27(8); publisher version may be paywalled). https://doi.org/10.1016/j.patrec.2005.10.010
  *Read for:* a practical primer on ROC curves and AUC. *Level:* intro.
- **Takaya Saito, Marc Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** (2015, PLOS ONE 10(3):e0118432; open access). https://doi.org/10.1371/journal.pone.0118432
  *Read for:* why ROC can flatter classifiers on imbalanced data. *Level:* intermediate.
- **Alexandru Niculescu-Mizil, Rich Caruana, "Predicting Good Probabilities with Supervised Learning"** (2005, ICML). https://doi.org/10.1145/1102351.1102430
  *Read for:* how learners distort probabilities, and Platt and isotonic repairs. *Level:* intermediate.
- **Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks"** (2017, ICML; arXiv:1706.04599). https://arxiv.org/abs/1706.04599
  *Read for:* overconfidence in deep networks and temperature scaling. *Level:* intermediate.

### L1-02: Linear and Logistic Models

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer). https://www.statlearning.com/
  *Read for:* the clearest gentle treatment of linear and logistic regression and ridge/lasso, with Python labs; free PDF on the official site. *Level:* intro.
- **scikit-learn developers, "1.1. Linear Models"** (scikit-learn 1.9.1 User Guide, accessed October 2026). https://scikit-learn.org/stable/modules/linear_model.html
  *Read for:* the exact objectives and solvers behind `Ridge`, `Lasso`, `ElasticNet`, and `LogisticRegression`. *Level:* intro.
- **scikit-learn developers, "Common pitfalls in the interpretation of coefficients of linear models"** (scikit-learn 1.9.1 examples, accessed October 2026). https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
  *Read for:* a real-data tour of units, conditional dependence, correlated features, coefficient variability, and causality. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani, Jerome Friedman, "The Elements of Statistical Learning: Data Mining, Inference, and Prediction"** (2nd ed., 2009, Springer). https://hastie.su.domains/ElemStatLearn/
  *Read for:* the SVD view of ridge, lasso geometry, and a deeper statistical treatment; free PDF on the authors' site. *Level:* intermediate.
- **Christopher M. Bishop, "Pattern Recognition and Machine Learning"** (2006, Springer). https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
  *Read for:* the probabilistic and Bayesian view of linear and logistic regression; free PDF from Microsoft Research. *Level:* intermediate.
- **Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction"** (2022, MIT Press). https://probml.github.io/pml-book/book1.html
  *Read for:* a modern probabilistic treatment of linear models, regularization as MAP estimation, and optimization; free draft PDF (CC BY-NC-ND). *Level:* intermediate.
- **Arthur E. Hoerl, Robert W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"** (1970, Technometrics 12(1):55-67). https://www.stat.cmu.edu/technometrics/70-79/VOL-12-01/v1201055.pdf
  *Read for:* the original case for adding a constant to the diagonal of $X^\top X$, and the "ridge trace". *Level:* intermediate.
- **Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso"** (1996, Journal of the Royal Statistical Society Series B 58(1):267-288). https://academic.oup.com/jrsssb/article/58/1/267/7027929
  *Read for:* the original lasso paper, including the orthonormal-design comparison with ridge. Publisher page; may require a subscription. *Level:* advanced.
- **Hui Zou, Trevor Hastie, "Regularization and variable selection via the elastic net"** (2005, Journal of the Royal Statistical Society Series B 67(2):301-320). https://ideas.repec.org/a/bla/jorssb/v67y2005i2p301-320.html
  *Read for:* why mixing L1 and L2 helps with correlated predictors and $p \gg n$. Bibliographic record; the journal version may require a subscription. *Level:* advanced.

### L1-03: Trees and Ensembles

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer). https://www.statlearning.com/
  *Read for:* a gentle chapter on trees, bagging, forests, and boosting with Python labs; free PDF. *Level:* intro.
- **scikit-learn developers, "Permutation Importance vs Random Forest Feature Importance (MDI)"** (scikit-learn 1.9.1 examples, accessed October 2026). https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
  *Read for:* a runnable demonstration of MDI's two main biases. *Level:* intro.
- **Leo Breiman, "Bagging Predictors"** (1996, Machine Learning 24:123-140). https://doi.org/10.1007/BF00058655
  *Read for:* the original bagging paper and why instability matters. Publisher page; may require a subscription. *Level:* intermediate.
- **Leo Breiman, "Random Forests"** (2001, Machine Learning 45(1):5-32). https://doi.org/10.1023/A:1010933404324
  *Read for:* random feature selection, OOB estimates, and the strength-correlation view. Publisher page; may require a subscription. *Level:* intermediate.
- **Jerome H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine"** (2001, Annals of Statistics 29(5):1189-1232). https://projecteuclid.org/euclid.aos/1013203451
  *Read for:* gradient boosting as steepest descent in function space, with algorithms for several losses. *Level:* advanced.
- **Tianqi Chen, Carlos Guestrin, "XGBoost: A Scalable Tree Boosting System"** (2016, KDD; arXiv:1603.02754). https://arxiv.org/abs/1603.02754
  *Read for:* the regularized second-order objective, leaf-weight and gain formulas, and systems design. *Level:* intermediate.
- **Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis, Torsten Hothorn, "Bias in random forest variable importance measures: Illustrations, sources and a solution"** (2007, BMC Bioinformatics 8:25). https://doi.org/10.1186/1471-2105-8-25
  *Read for:* why importance measures favor some predictor types, and a remedy; open access. *Level:* intermediate.
- **Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?"** (2022, NeurIPS Datasets and Benchmarks Track; arXiv:2207.08815). https://arxiv.org/abs/2207.08815
  *Read for:* a careful benchmark plus experiments on the inductive biases behind the gap. *Level:* intermediate.
- **Duncan McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"** (2023, NeurIPS Datasets and Benchmarks Track; arXiv:2305.02997). https://arxiv.org/abs/2305.02997
  *Read for:* a 176-dataset comparison of when each family wins, and by how little. *Level:* intermediate.
- **Noah Hollmann et al., "Accurate predictions on small data with a tabular foundation model"** (2025, Nature 637:319-326). https://doi.org/10.1038/s41586-024-08328-6
  *Read for:* TabPFN, a transformer pretrained on synthetic tables for small-data prediction. *Level:* advanced.
- **Nick Erickson et al., "TabArena: A Living Benchmark for Machine Learning on Tabular Data"** (2025, NeurIPS Datasets and Benchmarks Track; arXiv:2506.16791). https://arxiv.org/abs/2506.16791
  *Read for:* a maintained benchmark showing how validation and ensembling change rankings; check its live leaderboard. *Level:* intermediate.
- **Lennart Purucker et al. (including Gaël Varoquaux and Frank Hutter), "Beyond IID: How General Are Tabular Foundation Models, Really?"** (2026, arXiv:2606.30410, preprint). https://arxiv.org/abs/2606.30410
  *Read for:* where tabular foundation models stop leading (temporal, grouped, large, high-dimensional data). Preprint, not peer reviewed as of October 2026. *Level:* advanced.

### L1-04: Unsupervised Learning

- **Stuart P. Lloyd, "Least Squares Quantization in PCM"** (1982, IEEE Transactions on Information Theory 28(2), 129-137). <https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/LloydKMeans.pdf>
  *Read for:* the origin of Lloyd's algorithm in signal quantization (course copy; the IEEE version is paywalled). *Level:* advanced.
- **David Arthur and Sergei Vassilvitskii, "k-means++: The Advantages of Careful Seeding"** (2007, ACM-SIAM Symposium on Discrete Algorithms, SODA). <https://www.cs.columbia.edu/~verma/classes/uml/ref/clustering_kmean_plusplus_arthur_vassilvitskii.pdf>
  *Read for:* the $D^2$ seeding rule, its $O(\log k)$ guarantee, and experiments (course copy). *Level:* intermediate.
- **A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm"** (1977, Journal of the Royal Statistical Society Series B 39(1)). <https://academic.oup.com/jrsssb/article/39/1/1/7027539>
  *Read for:* the general EM framework and its monotone-likelihood property (publisher page, may be paywalled). *Level:* advanced.
- **Laurens van der Maaten and Geoffrey Hinton, "Visualizing Data using t-SNE"** (2008, Journal of Machine Learning Research 9, 2579-2605). <https://jmlr.org/papers/v9/vandermaaten08a.html>
  *Read for:* the t-SNE objective and why the Student-t kernel reduces crowding (open access). *Level:* intermediate.
- **Leland McInnes, John Healy and James Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"** (2018, arXiv:1802.03426). <https://arxiv.org/abs/1802.03426>
  *Read for:* UMAP's construction and its authors' comparison with t-SNE. *Level:* advanced.
- **Martin Wattenberg, Fernanda Viégas and Ian Johnson, "How to Use t-SNE Effectively"** (2016, Distill). <https://distill.pub/2016/misread-tsne/>
  *Read for:* interactive versions of every t-SNE reading rule here; the best first read. *Level:* intro.
- **Tara Chari and Lior Pachter, "The specious art of single-cell genomics"** (2023, PLOS Computational Biology 19(8): e1011288). <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288>
  *Read for:* a quantitative critique of t-SNE and UMAP as scientific evidence (open access; a correction notice was added in November 2025). *Level:* intermediate.
- **Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou, "Isolation Forest"** (2008, IEEE International Conference on Data Mining, 413-422). <https://doi.org/10.1109/ICDM.2008.17>
  *Read for:* the isolation idea and why it scales (IEEE, may be paywalled). *Level:* intermediate.
- **Peter J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"** (1987, Journal of Computational and Applied Mathematics 20, 53-65). <https://www.sciencedirect.com/science/article/pii/0377042787901257>
  *Read for:* the silhouette's definition and intended use (publisher page). *Level:* intermediate.
- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani and Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer; free PDF from the authors). <https://www.statlearning.com/>
  *Read for:* Chapter 12 (Unsupervised Learning), a gentle treatment of PCA and clustering with Python labs. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani and Jerome Friedman, "The Elements of Statistical Learning"** (2nd edition, 2009, Springer; free PDF from the authors). <https://hastie.su.domains/ElemStatLearn/>
  *Read for:* Chapter 14 (Unsupervised Learning), deeper on mixtures and clustering. *Level:* advanced.
- **scikit-learn developers, "Clustering", User Guide (version 1.9.1)**. <https://scikit-learn.org/stable/modules/clustering.html>
  *Read for:* algorithm comparisons, assumptions, and the ARI and silhouette metrics used here. *Level:* intro.

### L1-05: Data and Features

- **Nithya Sambasivan, Shivani Kapania, Hannah Highfill, Diana Akrong, Praveen Kumar Paritosh and Lora Mois Aroyo, "'Everyone wants to do the model work, not the data work': Data Cascades in High-Stakes AI"** (2021, ACM CHI). <https://research.google/pubs/pub49953/>
  *Read for:* interview evidence on how data problems compound in real projects. *Level:* intro.
- **Curtis G. Northcutt, Anish Athalye and Jonas Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"** (2021, NeurIPS Datasets and Benchmarks Track, arXiv:2103.14749). <https://arxiv.org/abs/2103.14749>
  *Read for:* measured label error rates in 10 benchmarks and how they change model rankings. *Level:* intermediate.
- **Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III and Kate Crawford, "Datasheets for Datasets"** (2021, Communications of the ACM; arXiv:1803.09010). <https://arxiv.org/abs/1803.09010>
  *Read for:* the full question list for documenting a dataset. *Level:* intro.
- **Donald B. Rubin, "Inference and Missing Data"** (1976, Biometrika 63(3), 581-592). <https://academic.oup.com/biomet/article/63/3/581/270932>
  *Read for:* the original conditions under which missingness can be ignored (paywalled). *Level:* advanced.
- **Stef van Buuren, "Flexible Imputation of Missing Data"** (2nd edition, 2018, Chapman & Hall/CRC; free online). <https://stefvanbuuren.name/fimd/>
  *Read for:* Section 1.2 on MCAR, MAR and MNAR, then multiple imputation in practice. *Level:* intermediate.
- **N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique"** (2002, Journal of Artificial Intelligence Research 16). <https://www.jair.org/index.php/jair/article/view/10302>
  *Read for:* the SMOTE algorithm and the original experiments (open access). *Level:* intermediate.
- **Ruben van den Goorbergh, Maarten van Smeden, Dirk Timmerman and Ben Van Calster, "The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression"** (2022, Journal of the American Medical Informatics Association 29(9), 1525-1534). <https://pmc.ncbi.nlm.nih.gov/articles/PMC9382395/>
  *Read for:* evidence that imbalance corrections damage calibration without improving discrimination (open access). *Level:* intermediate.
- **José G. Moreno-Torres, Troy Raeder, Rocío Alaiz-Rodríguez, Nitesh V. Chawla and Francisco Herrera, "A unifying view on dataset shift in classification"** (2012, Pattern Recognition 45(1), 521-530). <https://doi.org/10.1016/j.patcog.2011.06.019>
  *Read for:* a consistent vocabulary for the kinds of dataset shift (publisher version paywalled). *Level:* intermediate.
- **Zachary C. Lipton, Yu-Xiang Wang and Alex Smola, "Detecting and Correcting for Label Shift with Black Box Predictors"** (2018, ICML; arXiv:1802.03916). <https://arxiv.org/abs/1802.03916>
  *Read for:* BBSE, its assumptions and error bounds, and a test for label shift. *Level:* advanced.
- **D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems"** (2015, Advances in Neural Information Processing Systems 28). <https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  *Read for:* why ML systems are costly to maintain, with data dependencies and feedback loops as central causes. *Level:* intro.
- **Martin Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering"** (Google for Developers). <https://developers.google.com/machine-learning/guides/rules-of-ml>
  *Read for:* practical rules on features, logging and training-serving skew (Rules 29 to 37). *Level:* intro.
- **scikit-learn developers, "Preprocessing data", User Guide (version 1.9.1)**. <https://scikit-learn.org/stable/modules/preprocessing.html>
  *Read for:* scalers, encoders, `TargetEncoder` and its cross-fitting, with code. *Level:* intro.

## Level 2: Deep learning

### L2-01: Neural Networks and Backpropagation

- **Michael Nielsen, "Neural Networks and Deep Learning"** (free online book, 2015). <http://neuralnetworksanddeeplearning.com/>
  *Read for:* chapter 2 derives backpropagation in the same four-equation style as section 6, and chapter 4 gives a visual argument for universal approximation. *Level:* intro.
- **Grant Sanderson (3Blue1Brown), "Backpropagation calculus"** (Neural Networks series, chapter 5). <https://www.3blue1brown.com/lessons/backpropagation-calculus>
  *Read for:* the chain rule through a tiny network, visually, before the matrix version. *Level:* intro.
- **Stanford CS231n, "Neural Networks Part 3: Learning and Evaluation"** (course notes). <https://cs231n.github.io/neural-networks-3/>
  *Read for:* gradient-check practice (centered differences, relative error thresholds, double precision, kinks) and pre-training sanity checks. *Level:* intro.
- **Simon J. D. Prince, "Understanding Deep Learning"** (2023, MIT Press; free PDF from the author). <https://udlbook.github.io/udlbook/>
  *Read for:* chapter 3 on shallow networks and their piecewise-linear regions, and chapter 7 on gradients and initialization. *Level:* intro to intermediate.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML version). <https://www.deeplearningbook.org/>
  *Read for:* chapter 6, Deep Feedforward Networks: output units, hidden units, universal approximation and back-propagation. *Level:* intermediate.
- **Frank Rosenblatt, "The perceptron: A probabilistic model for information storage and organization in the brain"** (1958, Psychological Review 65(6), 386 to 408). <https://doi.org/10.1037/h0042519>
  *Read for:* the original perceptron paper, framed as a model of the brain. Mainly historical interest. *Level:* intermediate.
- **David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, "Learning representations by back-propagating errors"** (1986, Nature 323, 533 to 536; paywalled at the publisher). <https://doi.org/10.1038/323533a0>
  *Read for:* the short paper that showed backpropagation training hidden units to learn useful features. *Level:* intermediate.
- **George Cybenko, "Approximation by superpositions of a sigmoidal function"** (1989, Mathematics of Control, Signals and Systems 2, 303 to 314). <https://doi.org/10.1007/BF02551274>
  *Read for:* the classic one-hidden-layer universal approximation proof. *Level:* advanced.
- **Kurt Hornik, Maxwell Stinchcombe, Halbert White, "Multilayer feedforward networks are universal approximators"** (1989, Neural Networks 2(5), 359 to 366). <https://doi.org/10.1016/0893-6080(89)90020-8>
  *Read for:* universal approximation for arbitrary squashing functions. *Level:* advanced.
- **Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, Shimon Schocken, "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function"** (1993, Neural Networks 6(6), 861 to 867). <https://doi.org/10.1016/S0893-6080(05)80131-5>
  *Read for:* the clean "non-polynomial if and only if" condition, which covers ReLU. *Level:* advanced.
- **Matus Telgarsky, "Benefits of depth in neural networks"** (2016, COLT; arXiv:1602.04485). <https://arxiv.org/abs/1602.04485>
  *Read for:* the depth-separation result built on the sawtooth construction of script 2. *Level:* advanced.
- **Xavier Glorot, Antoine Bordes, Yoshua Bengio, "Deep Sparse Rectifier Neural Networks"** (2011, AISTATS, PMLR 15, 315 to 323). <https://proceedings.mlr.press/v15/glorot11a.html>
  *Read for:* evidence that ReLU networks train well without pre-training, and the case for sparse activations. *Level:* intermediate.

### L2-02: Training Deep Networks

- **Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola, "Dive into Deep Learning"** (2023, Cambridge University Press; free online, with PyTorch code). <https://d2l.ai/>
  *Read for:* chapter 6 (Builders' Guide) on modules, parameters and initialization, and section 12.11 on learning-rate scheduling. *Level:* intro.
- **Andrej Karpathy, "A Recipe for Training Neural Networks"** (2019, blog post). <https://karpathy.github.io/2019/04/25/recipe/>
  *Read for:* a step-by-step debugging and tuning process, including the initial-loss check and overfitting one batch. *Level:* intro.
- **PyTorch documentation: "Autograd mechanics", "Reproducibility", `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`, `nn.CrossEntropyLoss`** (version 2.15, checked October 2026). <https://docs.pytorch.org/docs/stable/notes/autograd.html>, <https://docs.pytorch.org/docs/stable/notes/randomness.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
  *Read for:* the exact contracts this module relies on: default initialization, running statistics, dropout scaling, logits in the loss, and what is and is not reproducible. *Level:* intro to intermediate.
- **Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, Zachary Nado, "Deep Learning Tuning Playbook"** (2023, GitHub, version 1). <https://github.com/google-research/tuning_playbook>
  *Read for:* a systematic process for tuning, including learning-rate decay choices and how to run comparisons. *Level:* intermediate.
- **Xavier Glorot, Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks"** (2010, AISTATS, PMLR 9, 249 to 256). <https://proceedings.mlr.press/v9/glorot10a.html>
  *Read for:* how activations and gradients behave across layers at initialization, and the Xavier scheme. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"** (2015, arXiv:1502.01852). <https://arxiv.org/abs/1502.01852>
  *Read for:* the derivation of the $2/n$ initialization for rectifier networks. *Level:* intermediate.
- **Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"** (2015, ICML, PMLR 37, 448 to 456; arXiv:1502.03167). <https://arxiv.org/abs/1502.03167>
  *Read for:* the original method, its train and inference procedures, and the internal-covariate-shift motivation. *Level:* intermediate.
- **Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization"** (2016, arXiv:1607.06450). <https://arxiv.org/abs/1607.06450>
  *Read for:* per-example normalization and why it suits recurrent networks. *Level:* intermediate.
- **Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry, "How Does Batch Normalization Help Optimization?"** (2018, NeurIPS; arXiv:1805.11604). <https://arxiv.org/abs/1805.11604>
  *Read for:* the argument that batch norm works by smoothing the loss landscape rather than by reducing covariate shift. *Level:* advanced.
- **Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** (2014, JMLR 15(56), 1929 to 1958). <https://jmlr.org/papers/v15/srivastava14a.html>
  *Read for:* the method, its interpretation as averaging many thinned networks, and extensive experiments. *Level:* intermediate.
- **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "Understanding deep learning requires rethinking generalization"** (2017, ICLR; arXiv:1611.03530). <https://arxiv.org/abs/1611.03530>
  *Read for:* the random-label experiments behind section 5, and why they unsettle classical explanations of generalization. *Level:* intermediate.
- **Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks"** (2017, WACV; arXiv:1506.01186). <https://arxiv.org/abs/1506.01186>
  *Read for:* the learning-rate range test and cyclical schedules. *Level:* intermediate.

### L2-03: Convolutional Networks

- **CS231n course staff, "Convolutional Neural Networks (CNNs / ConvNets)"** (Stanford CS231n notes, accessed October 2026). https://cs231n.github.io/convolutional-networks/
  *Read for:* conv arithmetic, parameter sharing, pooling, and case studies from LeNet to ResNet. *Level:* intro.
- **Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola, "Dive into Deep Learning"** (2023, arXiv:2106.11342; free online). https://d2l.ai/chapter_convolutional-neural-networks/
  *Read for:* CNN chapters with runnable PyTorch LeNet, VGG and ResNet. *Level:* intro.
- **Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, "Gradient-Based Learning Applied to Document Recognition"** (1998, Proceedings of the IEEE 86(11):2278-2324). https://doi.org/10.1109/5.726791
  *Read for:* LeNet and the case for learned features. Publisher copy may need a subscription; free copies are linked from https://bottou.org/papers/lecun-98h. *Level:* intermediate.
- **Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"** (2012, NeurIPS 25). https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
  *Read for:* the network that made large GPU-trained CNNs standard. *Level:* intermediate.
- **Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"** (2015, ICLR; arXiv:1409.1556). https://arxiv.org/abs/1409.1556
  *Read for:* the case for $3 \times 3$ stacks and a careful study of depth. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition"** (2016, CVPR; arXiv:1512.03385). https://arxiv.org/abs/1512.03385
  *Read for:* the degradation problem and residual blocks. Cited as [He et al., 2016a]. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks"** (2016, ECCV; arXiv:1603.05027). https://arxiv.org/abs/1603.05027
  *Read for:* gradient propagation through shortcuts and pre-activation blocks. Cited as [He et al., 2016b]. *Level:* advanced.
- **Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks"** (2016, NeurIPS 29; arXiv:1701.04128). https://arxiv.org/abs/1701.04128
  *Read for:* why the effective receptive field is Gaussian and small. *Level:* advanced.
- **Richard Zhang, "Making Convolutional Networks Shift-Invariant Again"** (2019, ICML; arXiv:1904.11486). https://arxiv.org/abs/1904.11486
  *Read for:* how pooling and striding break shift invariance, and the anti-aliasing fix. *Level:* intermediate.
- **Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson, "How transferable are features in deep neural networks?"** (2014, NeurIPS 27; arXiv:1411.1792). https://arxiv.org/abs/1411.1792
  *Read for:* layer-by-layer measurements of feature generality. *Level:* intermediate.
- **Kaiming He, Ross Girshick, Piotr Dollár, "Rethinking ImageNet Pre-training"** (2019, ICCV; arXiv:1811.08883). https://arxiv.org/abs/1811.08883
  *Read for:* when training from scratch matches pretraining. Cited as [He et al., 2019]. *Level:* intermediate.
- **Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, "A ConvNet for the 2020s"** (2022, CVPR; arXiv:2201.03545). https://arxiv.org/abs/2201.03545
  *Read for:* how far a modernised CNN goes against vision transformers. *Level:* advanced.

### L2-04: Sequence Models

- **Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks"** (2015, blog post). https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  *Read for:* character-level RNN language models, samples, a minimal NumPy implementation, and temperature. *Level:* intro.
- **Christopher Olah, "Understanding LSTM Networks"** (2015, blog post). https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  *Read for:* clear step-by-step diagrams of LSTM gates and variants. *Level:* intro.
- **Yoshua Bengio, Patrice Simard, Paolo Frasconi, "Learning Long-Term Dependencies with Gradient Descent is Difficult"** (1994, IEEE Transactions on Neural Networks 5:157ff). https://doi.org/10.1109/72.279181
  *Read for:* the original analysis of why long-range credit assignment is hard. May need a subscription. *Level:* advanced.
- **Sepp Hochreiter, Jürgen Schmidhuber, "Long Short-Term Memory"** (1997, Neural Computation 9(8):1735-1780). https://doi.org/10.1162/neco.1997.9.8.1735
  *Read for:* the original LSTM and its long-time-lag experiments. May need a subscription. *Level:* advanced.
- **Felix A. Gers, Jürgen Schmidhuber, Fred Cummins, "Learning to Forget: Continual Prediction with LSTM"** (2000, Neural Computation 12:2451-2471). https://doi.org/10.1162/089976600300015015
  *Read for:* why the forget gate was added. May need a subscription. *Level:* advanced.
- **Razvan Pascanu, Tomas Mikolov, Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks"** (2013, ICML, PMLR 28(3):1310-1318; arXiv:1211.5063). https://arxiv.org/abs/1211.5063
  *Read for:* the geometry of exploding and vanishing gradients, and norm clipping. *Level:* intermediate.
- **Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"** (2014, EMNLP; arXiv:1406.1078). https://arxiv.org/abs/1406.1078
  *Read for:* the encoder-decoder and the gated unit now called the GRU. *Level:* intermediate.
- **Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks"** (2014, NeurIPS 27; arXiv:1409.3215). https://arxiv.org/abs/1409.3215
  *Read for:* deep LSTM seq2seq, source reversal, and beam search in practice. *Level:* intermediate.
- **Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate"** (2015, ICLR; arXiv:1409.0473). https://arxiv.org/abs/1409.0473
  *Read for:* the origin of attention, with alignment plots. Cited as [Bahdanau et al., 2015]. *Level:* intermediate.
- **Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever, "An Empirical Exploration of Recurrent Network Architectures"** (2015, ICML, PMLR 37:2342-2350). https://proceedings.mlr.press/v37/jozefowicz15.html
  *Read for:* a large LSTM versus GRU comparison and the forget-bias finding. *Level:* intermediate.
- **Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"** (2015, NeurIPS 28; arXiv:1506.03099). https://arxiv.org/abs/1506.03099
  *Read for:* the train/test mismatch of teacher forcing and a curriculum fix. *Level:* intermediate.
- **Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi, "The Curious Case of Neural Text Degeneration"** (2020, ICLR; arXiv:1904.09751). https://arxiv.org/abs/1904.09751
  *Read for:* why likelihood-maximizing decoding degenerates, and nucleus sampling. *Level:* intermediate.

### L2-05: Transformers

- **Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need"** (2017, NeurIPS 30; arXiv:1706.03762). https://arxiv.org/abs/1706.03762
  *Read for:* the original architecture, the scaling argument, sinusoidal encodings and the complexity table. *Level:* intermediate.
- **Alexander (Sasha) Rush, "The Annotated Transformer"** (2018, Workshop for NLP Open Source Software; 2022 update by Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman). https://nlp.seas.harvard.edu/annotated-transformer/
  *Read for:* the paper rebuilt line by line in PyTorch. *Level:* intro to intermediate.
- **Simon J. D. Prince, "Understanding Deep Learning"** (2023, MIT Press). https://udlbook.github.io/udlbook/
  *Read for:* the transformers chapter's clear figures of attention and heads. Free PDF on the author's site. *Level:* intro.
- **Andrej Karpathy, "nanoGPT"** (GitHub repository, MIT license, 2022). https://github.com/karpathy/nanoGPT
  *Read for:* a compact GPT training codebase with a character-level Shakespeare example close to this module's. *Level:* intro to intermediate.
- **Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, "Language Models are Unsupervised Multitask Learners"** (2019, OpenAI report). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  *Read for:* GPT-2's pre-LN layout, scaled residual initialization and model sizes (section 2.3). *Level:* intermediate.
- **Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu, "On Layer Normalization in the Transformer Architecture"** (2020, ICML; arXiv:2002.04745). https://arxiv.org/abs/2002.04745
  *Read for:* why post-LN needs warmup and pre-LN does not. *Level:* advanced.
- **Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021, arXiv:2104.09864). https://arxiv.org/abs/2104.09864
  *Read for:* the derivation of rotary embeddings. *Level:* intermediate.
- **Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, Omer Levy, "Transformer Language Models without Positional Encodings Still Learn Positional Information"** (2022, Findings of EMNLP; arXiv:2203.16634). https://arxiv.org/abs/2203.16634
  *Read for:* evidence that the causal mask alone carries position information. *Level:* intermediate.
- **Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (2022, arXiv:2205.14135). https://arxiv.org/abs/2205.14135
  *Read for:* why attention speed is about memory traffic, and exact attention without the $T \times T$ matrix. *Level:* advanced.
- **PyTorch Contributors, "torch.nn.functional.scaled_dot_product_attention"** (PyTorch 2.15 documentation, accessed October 2026). https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
  *Read for:* the `is_causal` and `scale` arguments and the available backends. *Level:* intro.

### L2-06: Representation Learning and Embeddings

- **Daniel Jurafsky and James H. Martin, "Speech and Language Processing", 3rd edition draft, chapter 5 "Embeddings"** (draft of August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/
  *Read for:* the distributional hypothesis, PMI and skip-gram with negative sampling, explained for beginners. Free PDF. *Level:* intro.
- **Jay Alammar, "The Illustrated Word2vec"** (2019, blog). https://jalammar.github.io/illustrated-word2vec/
  *Read for:* a picture-first walk through skip-gram and negative sampling. *Level:* intro.
- **Zellig S. Harris, "Distributional Structure"** (1954, *Word* 10(2-3), 146 to 162). https://doi.org/10.1080/00437956.1954.11659520
  *Read for:* the original statement of the distributional hypothesis. Publisher version may be paywalled. *Level:* intermediate.
- **Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean, "Distributed Representations of Words and Phrases and their Compositionality"** (2013, NeurIPS 26; arXiv:1310.4546). https://arxiv.org/abs/1310.4546
  *Read for:* negative sampling, the $U^{3/4}$ noise distribution and subsampling of frequent words. *Level:* intermediate.
- **Omer Levy and Yoav Goldberg, "Neural Word Embedding as Implicit Matrix Factorization"** (2014, NeurIPS 27). https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html
  *Read for:* why SGNS is close to factorizing a shifted PMI matrix, and how SVD compares. *Level:* advanced.
- **Pierre Baldi and Kurt Hornik, "Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima"** (1989, *Neural Networks* 2(1), 53 to 58). https://doi.org/10.1016/0893-6080(89)90014-2
  *Read for:* the proof that linear autoencoders learn the PCA subspace. Publisher version may be paywalled. *Level:* advanced.
- **Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, "A Simple Framework for Contrastive Learning of Visual Representations"** (2020, ICML; arXiv:2002.05709). https://arxiv.org/abs/2002.05709
  *Read for:* SimCLR's ablations of augmentations, projection head, temperature and batch size. *Level:* intermediate.
- **Tongzhou Wang and Phillip Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"** (2020, ICML; arXiv:2005.10242). https://arxiv.org/abs/2005.10242
  *Read for:* the alignment and uniformity metrics used in script 03. *Level:* intermediate.
- **Miloš Radovanović, Alexandros Nanopoulos, Mirjana Ivanović, "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data"** (2010, *JMLR* 11, 2487 to 2531). https://jmlr.org/papers/v11/radovanovic10a.html
  *Read for:* hubness as a property of high-dimensional distributions. Open access. *Level:* advanced.
- **Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou, "The Faiss library"** (2024, arXiv:2401.08281). https://arxiv.org/abs/2401.08281
  *Read for:* the trade-offs of vector search (speed, memory, accuracy) and the indexing and compression methods in a widely used library. *Level:* intermediate.
- **Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, Adam Kalai, "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"** (2016, NeurIPS 29; arXiv:1607.06520). https://arxiv.org/abs/1607.06520
  *Read for:* gender bias as a direction in word2vec space, and a debiasing method. *Level:* intermediate.
- **Malvina Nissim, Rik van Noord, Rob van der Goot, "Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor"** (2020, *Computational Linguistics* 46(2), 487 to 497). https://aclanthology.org/2020.cl-2.7
  *Read for:* why analogy tests mislead as bias measurements. Open access. *Level:* intermediate.

## Level 3: Modern AI across domains

### L3-01: Language Model Pretraining

- **Rico Sennrich, Barry Haddow, Alexandra Birch, "Neural Machine Translation of Rare Words with Subword Units"** (2016, ACL; arXiv:1508.07909). https://arxiv.org/abs/1508.07909
  *Read for:* the original BPE-for-NLP algorithm, with a short reference implementation. *Level:* intro to intermediate.
- **Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, "Language Models are Unsupervised Multitask Learners"** (2019, OpenAI report). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  *Read for:* section 2.2 on byte-level BPE, and the case for next-token prediction as multitask learning. *Level:* intermediate.
- **Taku Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates"** (2018, ACL; arXiv:1804.10959). https://arxiv.org/abs/1804.10959
  *Read for:* the unigram language model tokenizer and sampled segmentations. *Level:* intermediate.
- **Andrej Karpathy, "minbpe"** (GitHub repository, MIT license). https://github.com/karpathy/minbpe
  *Read for:* compact byte-level BPE code whose regex tokenizer reproduces GPT-4's, with a linked lecture video. *Level:* intro.
- **Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei, "Scaling Laws for Neural Language Models"** (2020, arXiv:2001.08361). https://arxiv.org/abs/2001.08361
  *Read for:* the power laws that made extrapolation the basis of planning. *Level:* intermediate.
- **Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. (22 authors), "Training Compute-Optimal Large Language Models"** (2022, arXiv:2203.15556). https://arxiv.org/abs/2203.15556
  *Read for:* the three estimation methods, isoFLOP curves and the Chinchilla result. *Level:* intermediate to advanced.
- **Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You, "Chinchilla Scaling: A replication attempt"** (2024, arXiv:2404.10102). https://arxiv.org/abs/2404.10102
  *Read for:* a short, careful check of a famous fit, with corrected constants. *Level:* intermediate.
- **Jason Wei, Yi Tay, Rishi Bommasani, et al. (16 authors), "Emergent Abilities of Large Language Models"** (2022, TMLR; arXiv:2206.07682). https://arxiv.org/abs/2206.07682
  *Read for:* the definition and the catalogue of tasks that started the debate. *Level:* intro to intermediate.
- **Rylan Schaeffer, Brando Miranda, Sanmi Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?"** (2023, NeurIPS; arXiv:2304.15004). https://arxiv.org/abs/2304.15004
  *Read for:* the metric-choice argument and its simple mathematical model. *Level:* intro to intermediate.
- **Aaron Grattafiori et al. (Llama Team, Meta), "The Llama 3 Herd of Models"** (2024, arXiv:2407.21783). https://arxiv.org/abs/2407.21783
  *Read for:* the most detailed public account of a frontier-scale run: data, scaling-law use, context extension, annealing and failures at 16K GPUs. *Level:* intermediate to advanced.
- **Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"** (2024, arXiv:2406.17557). https://arxiv.org/abs/2406.17557
  *Read for:* each curation step with ablations, and the classifier behind FineWeb-Edu. *Level:* intro to intermediate.
- **Sebastian Raschka, "A Dream of Spring for Open-Weight LLMs: 10 Architectures from Jan-Feb 2026"** (Ahead of AI newsletter, February 25, 2026). https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
  *Read for:* side-by-side diagrams of recent open-weight architectures. A secondary source; check details against each technical report. *Level:* intro.

### L3-02: Post-Training and Alignment Methods

- **Nathan Lambert, "Reinforcement Learning from Human Feedback"** (book, free online at rlhfbook.com, print edition Manning 2026; arXiv:2504.12501). https://rlhfbook.com/
  *Read for:* a full, current treatment of every stage in this module, from reward models and policy gradients to direct alignment and RLVR. *Level:* intro to intermediate.
- **Long Ouyang, Jeff Wu, Xu Jiang, et al. (20 authors), "Training language models to follow instructions with human feedback"** (2022, arXiv:2203.02155). https://arxiv.org/abs/2203.02155
  *Read for:* the SFT, reward model and PPO pipeline and the human evaluations behind it. *Level:* intro to intermediate.
- **John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, "Proximal Policy Optimization Algorithms"** (2017, arXiv:1707.06347). https://arxiv.org/abs/1707.06347
  *Read for:* the clipped objective; read L3-05 first. *Level:* intermediate.
- **Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** (2023, NeurIPS; arXiv:2305.18290). https://arxiv.org/abs/2305.18290
  *Read for:* the derivation in section 4 and the implicit-reward view. *Level:* intermediate.
- **Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos, "A General Theoretical Paradigm to Understand Learning from Human Preferences"** (2023, arXiv:2310.12036). https://arxiv.org/abs/2310.12036
  *Read for:* why DPO can ignore its regularization, and the IPO loss. *Level:* advanced.
- **Leo Gao, John Schulman, Jacob Hilton, "Scaling Laws for Reward Model Overoptimization"** (2022, arXiv:2210.10760). https://arxiv.org/abs/2210.10760
  *Read for:* the gold-versus-proxy methodology and how over-optimization scales. *Level:* intermediate.
- **Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al. (51 authors), "Constitutional AI: Harmlessness from AI Feedback"** (2022, arXiv:2212.08073). https://arxiv.org/abs/2212.08073
  *Read for:* critique-and-revise SFT and RL from AI feedback. *Level:* intro to intermediate.
- **Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. (11 authors), "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"** (2024, arXiv:2402.03300). https://arxiv.org/abs/2402.03300
  *Read for:* the GRPO objective, its group-normalized advantage and KL estimator (section 4). *Level:* intermediate.
- **DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** (2025, Nature 645, 633-638; arXiv:2501.12948). https://arxiv.org/abs/2501.12948
  *Read for:* RL on a base model with rule-based rewards, what emerged, and the four-stage pipeline. *Level:* intermediate.
- **Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, et al. (22 authors), "Tulu 3: Pushing Frontiers in Open Language Model Post-Training"** (2024, arXiv:2411.15124). https://arxiv.org/abs/2411.15124
  *Read for:* a fully open SFT, DPO and RLVR recipe with data and ablations. *Level:* intermediate.
- **Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang, et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"** (2025, NeurIPS; arXiv:2504.13837). https://arxiv.org/abs/2504.13837
  *Read for:* the pass@$k$ methodology and the case that RLVR mostly sharpens. *Level:* intermediate.
- **Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, Noah A. Smith, Yejin Choi, Hannaneh Hajishirzi, "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback"** (2024, NeurIPS; arXiv:2406.09279). https://arxiv.org/abs/2406.09279
  *Read for:* controlled comparisons showing data quality matters more than the algorithm. *Level:* intermediate.

### L3-03: Generative Models and Diffusion

- **Lilian Weng, "What are Diffusion Models?"** (2021, updated 2022 and 2024, blog). https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  *Read for:* a careful, derivation-level tour of DDPM, score matching connections, DDIM, guidance, latent diffusion, distillation and consistency models. *Level:* intro.
- **Calvin Luo, "Understanding Diffusion Models: A Unified Perspective"** (2022, arXiv:2208.11970). https://arxiv.org/abs/2208.11970
  *Read for:* the ELBO view of diffusion as a hierarchical VAE, the three equivalent prediction targets, and Tweedie's formula linking them to scores. *Level:* intro to intermediate.
- **Diederik P. Kingma and Max Welling, "Auto-Encoding Variational Bayes"** (2014, ICLR; arXiv:1312.6114). https://arxiv.org/abs/1312.6114
  *Read for:* the ELBO, the reparameterization trick and the original VAE. *Level:* intermediate.
- **Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, "Generative Adversarial Nets"** (2014, NeurIPS 27; arXiv:1406.2661). https://arxiv.org/abs/1406.2661
  *Read for:* the minimax game, the optimal discriminator and the non-saturating generator loss. *Level:* intermediate.
- **Jonathan Ho, Ajay Jain, Pieter Abbeel, "Denoising Diffusion Probabilistic Models"** (2020, NeurIPS 33; arXiv:2006.11239). https://arxiv.org/abs/2006.11239
  *Read for:* the forward process, the simplified noise-prediction loss, the schedule used in this module, and an honest discussion of likelihoods. *Level:* intermediate.
- **Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole, "Score-Based Generative Modeling through Stochastic Differential Equations"** (2021, ICLR; arXiv:2011.13456). https://arxiv.org/abs/2011.13456
  *Read for:* the SDE view, reverse-time sampling with the score, and the probability flow ODE. *Level:* advanced.
- **Jonathan Ho and Tim Salimans, "Classifier-Free Diffusion Guidance"** (2022, arXiv:2207.12598; short version at a NeurIPS 2021 workshop). https://arxiv.org/abs/2207.12598
  *Read for:* the guidance formula, label dropout, and the FID versus Inception-score trade-off. Short and readable. *Level:* intermediate.
- **Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models"** (2022, CVPR; arXiv:2112.10752). https://arxiv.org/abs/2112.10752
  *Read for:* diffusion in an autoencoder's latent space, the choice of compression factor, and cross-attention conditioning. *Level:* intermediate.
- **Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, "Flow Matching for Generative Modeling"** (2023, ICLR; arXiv:2210.02747). https://arxiv.org/abs/2210.02747
  *Read for:* the conditional flow matching objective, why it recovers the marginal flow, and diffusion as a special case. *Level:* advanced.
- **Patrick Esser, Sumith Kulal, Andreas Blattmann, et al. (17 authors), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"** (2024, ICML; arXiv:2403.03206). https://arxiv.org/abs/2403.03206
  *Read for:* a large comparison of diffusion and rectified-flow formulations, time sampling for training, and the transformer used in Stable Diffusion 3. *Level:* advanced.
- **Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"** (2017, NeurIPS 30; arXiv:1706.08500). https://arxiv.org/abs/1706.08500
  *Read for:* the definition of FID and the case made for it. *Level:* intermediate.
- **George Stein, Jesse C. Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L. Caterini, J. Eric T. Taylor, Gabriel Loaiza-Ganem, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models"** (2023, NeurIPS; arXiv:2306.04675). https://arxiv.org/abs/2306.04675
  *Read for:* a large human study against 17 metrics, the case against Inception features, and why memorization is hard to measure. *Level:* intermediate.

### L3-04: Vision and Multimodal Models

- **Chip Huyen, "Multimodality and Large Multimodal Models (LMMs)"** (2023, blog). https://huyenchip.com/2023/10/10/multimodal.html
  *Read for:* an accessible overview of multimodal tasks, CLIP and Flamingo, and research directions. *Level:* intro.
- **merve and Edward Beeching, "Vision Language Models Explained"** (Hugging Face blog, 2024). https://huggingface.co/blog/vlms
  *Read for:* the encoder, projector and decoder structure of open VLMs, LLaVA-style training stages, and VLM benchmarks. *Level:* intro.
- **Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** (2021, ICLR; arXiv:2010.11929). https://arxiv.org/abs/2010.11929
  *Read for:* the ViT architecture and the evidence on data scale versus inductive bias. *Level:* intermediate.
- **Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, "End-to-End Object Detection with Transformers"** (2020, ECCV; arXiv:2005.12872). https://arxiv.org/abs/2005.12872
  *Read for:* detection as set prediction with bipartite matching, without NMS or anchors. *Level:* intermediate.
- **Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick, "Segment Anything"** (2023, ICCV; arXiv:2304.02643). https://arxiv.org/abs/2304.02643
  *Read for:* promptable segmentation and building a billion-mask dataset with a model in the loop. *Level:* intermediate.
- **Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, "Learning Transferable Visual Models From Natural Language Supervision"** (2021, ICML; arXiv:2103.00020). https://arxiv.org/abs/2103.00020
  *Read for:* the CLIP objective, zero-shot transfer, prompt engineering, and an unusually thorough limitations section. *Level:* intermediate.
- **Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou, "When and why vision-language models behave like bags-of-words, and what to do about it?"** (2023, ICLR; arXiv:2210.01936). https://arxiv.org/abs/2210.01936
  *Read for:* the ARO benchmark and composition-aware hard negatives. *Level:* intermediate.
- **Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee, "Visual Instruction Tuning"** (2023, NeurIPS; arXiv:2304.08485). https://arxiv.org/abs/2304.08485
  *Read for:* LLaVA, connecting a vision encoder to an LLM and generating instruction data with a text-only model. *Level:* intermediate.
- **Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models"** (2024, arXiv:2405.09818). https://arxiv.org/abs/2405.09818
  *Read for:* early fusion with discrete image tokens and the training-stability issues it raises. *Level:* advanced.
- **Qwen Team, "Qwen3-VL Technical Report"** (2025, arXiv:2511.21631). https://arxiv.org/abs/2511.21631
  *Read for:* a detailed, recent account of a large open VLM: native-resolution encoder, token merging, multi-level feature injection and positions for video. *Level:* advanced.
- **Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie, "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"** (2024, CVPR; arXiv:2401.06209). https://arxiv.org/abs/2401.06209
  *Read for:* CLIP-blind image pairs and the MMVP benchmark of basic visual failures in VLMs. *Level:* intermediate.
- **Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision"** (2023, ICML; arXiv:2212.04356). https://arxiv.org/abs/2212.04356
  *Read for:* Whisper's data, front end, multitask token format and its documented failure modes. *Level:* intermediate.

### L3-05: Reinforcement Learning Foundations

- **Richard S. Sutton and Andrew G. Barto, "Reinforcement Learning: An Introduction", 2nd edition** (2018, MIT Press). http://incompleteideas.net/book/the-book-2nd.html
  *Read for:* the standard text, including the random-walk and cliff-walking examples used here. Free PDF from the authors. *Level:* intro.
- **OpenAI, "Spinning Up in Deep RL", Part 3: Intro to Policy Optimization** (documentation). https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
  *Read for:* a short derivation of the policy gradient, reward-to-go and baselines, with code. *Level:* intro.
- **Alex Irpan, "Deep Reinforcement Learning Doesn't Work Yet"** (2018, blog). https://www.alexirpan.com/2018/02/14/rl-hard.html
  *Read for:* concrete examples of sample inefficiency, reward hacking, local optima and seed variance. *Level:* intro.
- **Christopher J. C. H. Watkins and Peter Dayan, "Q-learning"** (1992, *Machine Learning* 8, 279 to 292). https://doi.org/10.1007/BF00992698
  *Read for:* the convergence proof for tabular Q-learning. May be paywalled. *Level:* advanced.
- **Ronald J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"** (1992, *Machine Learning* 8, 229 to 256). https://doi.org/10.1007/BF00992696
  *Read for:* the original REINFORCE algorithms and reinforcement baselines. May be paywalled. *Level:* advanced.
- **Volodymyr Mnih et al., "Human-level control through deep reinforcement learning"** (2015, *Nature* 518, 529 to 533). https://doi.org/10.1038/nature14236
  *Read for:* DQN with replay and target networks on 49 Atari games. Paywalled; the earlier workshop paper "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013, arXiv:1312.5602, https://arxiv.org/abs/1312.5602) is free. *Level:* intermediate.
- **Hado van Hasselt, Arthur Guez, David Silver, "Deep Reinforcement Learning with Double Q-learning"** (2016, AAAI; arXiv:1509.06461). https://arxiv.org/abs/1509.06461
  *Read for:* why max-based targets overestimate, and the Double DQN fix. *Level:* intermediate.
- **John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel, "Trust Region Policy Optimization"** (2015, ICML; arXiv:1502.05477). https://arxiv.org/abs/1502.05477
  *Read for:* the KL-constrained update that PPO approximates. *Level:* advanced.
- **John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation"** (2016, ICLR; arXiv:1506.02438). https://arxiv.org/abs/1506.02438
  *Read for:* GAE and the bias-variance role of $\lambda$. *Level:* advanced.
- **John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, "Proximal Policy Optimization Algorithms"** (2017, arXiv:1707.06347). https://arxiv.org/abs/1707.06347
  *Read for:* the clipped objective, the KL-penalty variant, and Table 1's comparison of clip values. *Level:* intermediate.
- **Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang, "The 37 Implementation Details of Proximal Policy Optimization"** (2022, ICLR Blog Track). https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  *Read for:* what reference PPO implementations actually do, and which details matter. *Level:* intermediate.
- **Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger, "Deep Reinforcement Learning that Matters"** (2018, AAAI; arXiv:1709.06560). https://arxiv.org/abs/1709.06560
  *Read for:* how seeds, hyperparameters and implementations change deep RL results, and how to report them. *Level:* intermediate.

### L3-06: Retrieval and RAG

- **Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, "Introduction to Information Retrieval"** (2008, Cambridge University Press). https://nlp.stanford.edu/IR-book/
  *Read for:* inverted indexes, tf-idf scoring, evaluation of ranked retrieval and latent semantic indexing, explained from scratch. Free online edition. *Level:* intro.
- **Daniel Jurafsky and James H. Martin, "Speech and Language Processing", 3rd edition draft, chapter 11 "Information Retrieval and RAG"** (draft of August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/
  *Read for:* a current textbook treatment of sparse and dense retrieval and RAG. Free PDF. *Level:* intro.
- **Stephen Robertson and Hugo Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"** (2009, *Foundations and Trends in Information Retrieval* 3(4), 333 to 389). https://doi.org/10.1561/1500000019
  *Read for:* where BM25 comes from and what $k_1$ and $b$ mean. A free author copy is at https://staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf. *Level:* advanced.
- **Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, "Dense Passage Retrieval for Open-Domain Question Answering"** (2020, EMNLP; arXiv:2004.04906). https://arxiv.org/abs/2004.04906
  *Read for:* training a bi-encoder with in-batch negatives, and its comparison with BM25. *Level:* intermediate.
- **Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (2020, NeurIPS; arXiv:2005.11401). https://arxiv.org/abs/2005.11401
  *Read for:* the original RAG model and its parametric versus non-parametric memory framing. *Level:* intermediate.
- **Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"** (2009, SIGIR, 758 to 759). https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  *Read for:* the two-page paper behind RRF. *Level:* intro.
- **Yu. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"** (2020, *IEEE TPAMI* 42(4), 824 to 836; arXiv:1603.09320). https://arxiv.org/abs/1603.09320
  *Read for:* HNSW's layers, the $m_L = 1/\ln M$ choice, and the neighbor-selection heuristic. *Level:* advanced.
- **Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych, "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2104.08663). https://arxiv.org/abs/2104.08663
  *Read for:* how sparse, dense, late-interaction and reranking models generalize out of domain. *Level:* intermediate.
- **Omar Khattab and Matei Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"** (2020, SIGIR; arXiv:2004.12832). https://arxiv.org/abs/2004.12832
  *Read for:* late interaction as a middle ground between bi-encoders and cross-encoders. *Level:* advanced.
- **Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang, "Lost in the Middle: How Language Models Use Long Contexts"** (2024, TACL; arXiv:2307.03172). https://arxiv.org/abs/2307.03172
  *Read for:* the position effect in long contexts and what it means for ordering retrieved passages. *Level:* intermediate.
- **Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky, "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach"** (2024, EMNLP Industry Track; arXiv:2407.16833). https://arxiv.org/abs/2407.16833
  *Read for:* a controlled comparison of RAG and long context, and the Self-Route router. *Level:* intermediate.
- **Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert, "Ragas: Automated Evaluation of Retrieval Augmented Generation"** (2023, arXiv:2309.15217). https://arxiv.org/abs/2309.15217
  *Read for:* reference-free metrics for context quality, faithfulness and answer quality, and their assumptions. *Level:* intermediate.

### L3-07: Agents and Tool Use

- **"AI Systems" course, Track A: "Agents, tools and MCP"** (course website; source at github.com/sankarsh98/ai-systems; accessed October 2026). https://sankarsh98.github.io/ai-systems/
  *Read for:* the tool-use loop, tool design, MCP and everyday pitfalls, with runnable code; the prerequisite for this module. *Level:* intro.
- **Erik S. and Barry Zhang, "Building effective agents"** (Anthropic Engineering, December 2024). https://www.anthropic.com/engineering/building-effective-agents
  *Read for:* workflows versus agents, five workflow patterns, and the case for starting simple. The byline gives only the first author's initial. *Level:* intro.
- **Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe, "Demystifying evals for AI agents"** (Anthropic Engineering, January 2026). https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
  *Read for:* eval vocabulary, pass@k versus pass^k, outcome grading, capability versus regression suites, reading transcripts. *Level:* intro.
- **Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao, "ReAct: Synergizing Reasoning and Acting in Language Models"** (2023, ICLR; arXiv:2210.03629). https://arxiv.org/abs/2210.03629
  *Read for:* the reasoning-plus-acting loop that most agents still resemble. *Level:* intermediate.
- **Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan, "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"** (2024, arXiv:2406.12045). https://arxiv.org/abs/2406.12045
  *Read for:* grading against a goal database state, and the pass^k reliability metric. *Level:* intermediate.
- **Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan, "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"** (2024, ICLR; arXiv:2310.06770). https://arxiv.org/abs/2310.06770
  *Read for:* how the most cited coding-agent benchmark was built; read it with the critiques in section 7. *Level:* intermediate.
- **Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, Arvind Narayanan, "AI Agents That Matter"** (2024, arXiv:2407.01502). https://arxiv.org/abs/2407.01502
  *Read for:* cost-aware evaluation, holdouts and reproducibility in agent benchmarks. *Level:* intermediate.
- **Yuxuan Zhu et al. (25 authors), "Establishing Best Practices for Building Rigorous Agentic Benchmarks"** (2025, arXiv:2507.02825). https://arxiv.org/abs/2507.02825
  *Read for:* concrete flaws in popular agent benchmarks and the Agentic Benchmark Checklist. *Level:* intermediate.
- **Thomas Kwa, Ben West, Joel Becker, Amy Deng, et al. (26 authors), "Measuring AI Ability to Complete Long Software Tasks"** (2025, NeurIPS; arXiv:2503.14499). https://arxiv.org/abs/2503.14499
  *Read for:* the 50% time-horizon metric and its trend; pair it with METR's January 2026 note on limitations. *Level:* intermediate.
- **Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz, "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection"** (2023, arXiv:2302.12173). https://arxiv.org/abs/2302.12173
  *Read for:* the original demonstration and taxonomy of indirect prompt injection. *Level:* intermediate.
- **Luca Beurer-Kellner, Beat Buesser, Ana-Maria Creţu, Edoardo Debenedetti, et al. (14 authors), "Design Patterns for Securing LLM Agents against Prompt Injections"** (2025, arXiv:2506.08837). https://arxiv.org/abs/2506.08837
  *Read for:* six architectural patterns with their guarantees and utility costs. *Level:* intermediate.
- **Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr, "Defeating Prompt Injections by Design"** (2025, arXiv:2503.18813). https://arxiv.org/abs/2503.18813
  *Read for:* CaMeL, which separates control and data flow and enforces policies with capabilities. *Level:* advanced.

### L3-08: Recommenders and Ranking

- **Google for Developers, "Recommendation Systems"** (course, last updated August 2025). https://developers.google.com/machine-learning/recommendation
  *Read for:* a short, free introduction to candidate generation, scoring and re-ranking, matrix factorization and softmax models. *Level:* intro.
- **Yehuda Koren, Robert Bell, Chris Volinsky, "Matrix Factorization Techniques for Recommender Systems"** (2009, IEEE Computer 42(8), 30 to 37). https://doi.org/10.1109/MC.2009.263
  *Read for:* biased MF, SGD and ALS, and lessons from the Netflix Prize, in a readable magazine article. Publisher version may be paywalled; author copies circulate. *Level:* intro to intermediate.
- **Yifan Hu, Yehuda Koren, Chris Volinsky, "Collaborative Filtering for Implicit Feedback Datasets"** (2008, IEEE ICDM, 263 to 272). https://doi.org/10.1109/ICDM.2008.22
  *Read for:* preference and confidence, the weighted objective and the efficient ALS trick. Publisher version may be paywalled. *Level:* intermediate.
- **Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback"** (2009, UAI; arXiv:1205.2618). https://arxiv.org/abs/1205.2618
  *Read for:* the pairwise ranking objective for implicit data. *Level:* intermediate.
- **Paul Covington, Jay Adams, Emre Sargin, "Deep Neural Networks for YouTube Recommendations"** (2016, ACM RecSys, 191 to 198). https://doi.org/10.1145/2959100.2959190
  *Read for:* the candidate generation plus ranking design and many practical lessons. Free PDF on Google Research. *Level:* intermediate.
- **Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"** (2019, ACM RecSys, 269 to 277). https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
  *Read for:* two-tower retrieval with in-batch softmax, the logQ correction and streaming frequency estimation. *Level:* advanced.
- **Walid Krichene and Steffen Rendle, "On Sampled Metrics for Item Recommendation"** (2020, ACM KDD, 1748 to 1757). https://doi.org/10.1145/3394486.3403226
  *Read for:* why sampled ranking metrics mislead, and corrections. *Level:* intermediate.
- **Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"** (2019, ACM RecSys; arXiv:1907.06902). https://arxiv.org/abs/1907.06902
  *Read for:* a reproducibility study showing the value of tuned simple baselines. *Level:* intermediate.
- **Steffen Rendle, Walid Krichene, Li Zhang, John Anderson, "Neural Collaborative Filtering vs. Matrix Factorization Revisited"** (2020, arXiv:2005.09683). https://arxiv.org/abs/2005.09683
  *Read for:* why a tuned dot product beats learned MLP similarity and why dot products matter for retrieval. *Level:* intermediate.
- **Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, Thorsten Joachims, "Recommendations as Treatments: Debiasing Learning and Evaluation"** (2016, ICML; arXiv:1602.05352). https://arxiv.org/abs/1602.05352
  *Read for:* propensity-weighted learning and evaluation under selection bias. *Level:* advanced.
- **Allison J.B. Chaney, Brandon M. Stewart, Barbara E. Engelhardt, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility"** (2018, ACM RecSys; arXiv:1710.11214). https://arxiv.org/abs/1710.11214
  *Read for:* simulations of the feedback loop that script 04 imitates. *Level:* intermediate.
- **Andrew M. Guess, Neil Malhotra, Jennifer Pan, Pablo Barberá, et al., "How do social media feed algorithms affect attitudes and behavior in an election campaign?"** (2023, Science 381, 398 to 404). https://doi.org/10.1126/science.abp9364
  *Read for:* a large field experiment on algorithmic versus chronological feeds; read with the Bagchi et al. eLetter. May be paywalled. *Level:* intermediate.

### L3-09: Time Series and Graphs

- **Rob J Hyndman, George Athanasopoulos, "Forecasting: Principles and Practice", 3rd edition** (OTexts; free online, checked October 2026). https://otexts.com/fpp3/
  *Read for:* baselines, decomposition, ETS (ch. 8), ARIMA (ch. 9) and evaluation, with R code. *Level:* intro.
- **Benjamin Sanchez-Lengeling, Emily Reif, Adam Pearce, Alexander B. Wiltschko, "A Gentle Introduction to Graph Neural Networks"** (2021, Distill). https://distill.pub/2021/gnn-intro/
  *Read for:* an interactive visual introduction to graphs and message passing. *Level:* intro.
- **Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods"** (2020, International Journal of Forecasting 36(1):54-74; may be paywalled). https://doi.org/10.1016/j.ijforecast.2019.04.014
  *Read for:* how a large forecasting competition is run and what won. *Level:* intermediate.
- **Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions"** (2022, International Journal of Forecasting 38(4):1346-1364; may be paywalled). https://doi.org/10.1016/j.ijforecast.2021.11.013
  *Read for:* why global LightGBM models and combinations won, and how few teams beat the benchmarks. *Level:* intermediate.
- **Rob J. Hyndman, Anne B. Koehler, "Another look at measures of forecast accuracy"** (2006, International Journal of Forecasting 22(4):679-688; may be paywalled). https://doi.org/10.1016/j.ijforecast.2006.03.001
  *Read for:* why percentage errors mislead and the case for MASE. *Level:* intro.
- **Christoph Bergmeir, Rob J. Hyndman, Bonsoo Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction"** (2018, Computational Statistics & Data Analysis 120:70-83). https://doi.org/10.1016/j.csda.2017.11.003
  *Read for:* exactly when K-fold CV is valid for time series. *Level:* advanced.
- **Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, "Are Transformers Effective for Time Series Forecasting?"** (2023, AAAI; arXiv:2205.13504). https://arxiv.org/abs/2205.13504
  *Read for:* a linear model beating Transformer forecasters, and a lesson in baselines. *Level:* intermediate.
- **Abdul Fatir Ansari et al., "Chronos: Learning the Language of Time Series"** (2024, Transactions on Machine Learning Research; arXiv:2403.07815). https://arxiv.org/abs/2403.07815
  *Read for:* how a time-series foundation model is built and evaluated, in its authors' words. *Level:* advanced.
- **Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, Yuyang Wang, "fev-bench: A Realistic Benchmark for Time Series Forecasting"** (2025, arXiv:2509.26468, v4 June 2026). https://arxiv.org/abs/2509.26468
  *Read for:* a current benchmark with covariates, intervals and leakage reporting; read its limitations too. *Level:* advanced.
- **Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection"** (2022, AAAI; arXiv:2109.05257). https://arxiv.org/abs/2109.05257
  *Read for:* why point adjustment inflates scores, and the PA%K alternative. *Level:* intermediate.
- **William L. Hamilton, "Graph Representation Learning"** (2020, Morgan & Claypool, Synthesis Lectures on AI and ML 14(3); free pre-publication PDF). https://www.cs.mcgill.ca/~wlh/grl_book/
  *Read for:* a compact, rigorous book on message passing and GNN theory. *Level:* intermediate.
- **Thomas N. Kipf, Max Welling, "Semi-Supervised Classification with Graph Convolutional Networks"** (2017, ICLR; arXiv:1609.02907). https://arxiv.org/abs/1609.02907
  *Read for:* the GCN layer and the semi-supervised setting of script 5. *Level:* intermediate.

### L3-10: AI for Science and Healthcare

- **John Jumper et al., "Highly accurate protein structure prediction with AlphaFold"** (2021, Nature 596:583-589; free via PubMed Central). https://doi.org/10.1038/s41586-021-03819-2
  *Read for:* the AlphaFold2 method, CASP14 results and stated limitations. *Level:* advanced.
- **Thomas C. Terwilliger et al., "AlphaFold predictions are valuable hypotheses and accelerate but do not replace experimental structure determination"** (2024, Nature Methods 21:110-116; free via PubMed Central). https://doi.org/10.1038/s41592-023-02087-4
  *Read for:* an independent, quantitative check of AlphaFold against experimental maps. *Level:* intermediate.
- **Remi Lam et al., "Learning skillful medium-range global weather forecasting"** (2023, Science 382:1416-1421; arXiv:2212.12794 is free). https://arxiv.org/abs/2212.12794
  *Read for:* GraphCast, a graph neural network evaluated against operational forecasts. *Level:* advanced.
- **Ilan Price et al., "Probabilistic weather forecasting with machine learning"** (2025, Nature 637:84-90; free via PubMed Central). https://doi.org/10.1038/s41586-024-08252-9
  *Read for:* GenCast, ensemble evaluation, and a frank limitations section. *Level:* advanced.
- **Anthony K. Cheetham, Ram Seshadri, "Artificial Intelligence Driving Materials Discovery? Perspective on the Article: Scaling Deep Learning for Materials Discovery"** (2024, Chemistry of Materials 36:3490-3495; free via PubMed Central). https://doi.org/10.1021/acs.chemmater.4c00643
  *Read for:* how domain experts test a large discovery claim. *Level:* intermediate.
- **Josh Leeman et al., "Challenges in High-Throughput Inorganic Materials Prediction and Autonomous Synthesis"** (2024, PRX Energy 3:011002; open access). https://doi.org/10.1103/PRXEnergy.3.011002
  *Read for:* what "a new material" requires, and the A-Lab critique. *Level:* intermediate.
- **John R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study"** (2018, PLOS Medicine 15(11):e1002683; open access). https://doi.org/10.1371/journal.pmed.1002683
  *Read for:* the clearest worked example of site shortcuts in imaging. *Level:* intro.
- **Andrew Wong et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients"** (2021, JAMA Internal Medicine 181(8):1065-1070; open access via PMC). https://doi.org/10.1001/jamainternmed.2021.2626
  *Read for:* what independent external validation of a deployed model looks like. *Level:* intro.
- **Ziad Obermeyer, Brian Powers, Christine Vogeli, Sendhil Mullainathan, "Dissecting racial bias in an algorithm used to manage the health of populations"** (2019, Science 366:447-453; may be paywalled). https://doi.org/10.1126/science.aax2342
  *Read for:* how a proxy label creates bias, found by auditing outcomes by group. *Level:* intro.
- **Jessie Gommers et al., "Interval cancer, sensitivity, and specificity comparing AI-supported mammography screening with standard double reading without AI in the MASAI study"** (2026, The Lancet 407:505-514; may be paywalled). https://doi.org/10.1016/S0140-6736(25)02464-X
  *Read for:* a large randomised trial of clinical AI and how to read a non-inferiority result. *Level:* intermediate.
- **Robert Geirhos et al., "Shortcut learning in deep neural networks"** (2020, Nature Machine Intelligence 2:665; arXiv:2004.07780). https://doi.org/10.1038/s42256-020-00257-z
  *Read for:* a general account of shortcut learning across fields. *Level:* intro.
- **Gary S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods"** (2024, BMJ 385:e078378; open access). https://doi.org/10.1136/bmj-2023-078378
  *Read for:* the checklist a clinical prediction paper should satisfy. *Level:* intermediate.

## Level 4: Engineering and society

### L4-01: MLOps and Deployment

- **D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison, "Hidden Technical Debt in Machine Learning Systems"** (2015, Advances in Neural Information Processing Systems 28). <https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  *Read for:* why ML systems are expensive to maintain. *Level:* intro.
- **Google Cloud (Jarek Kazmierczak, Khalid Salama, Valentín Huerta, et al.), "MLOps: Continuous delivery and automation pipelines in machine learning"** (architecture guide, last updated August 28, 2024). <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
  *Read for:* MLOps levels 0 to 2, continuous training and retraining triggers. *Level:* intro.
- **Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley, "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"** (2017, IEEE Big Data). <https://research.google/pubs/pub46555/>
  *Read for:* 28 concrete tests for data, models, infrastructure and monitoring, and the scoring rule. *Level:* intro.
- **Eric Breck, Marty Zinkevich, Neoklis Polyzotis, Steven Whang, Sudip Roy, "Data Validation for Machine Learning"** (2019, SysML/MLSys). <https://research.google/pubs/data-validation-for-machine-learning/>
  *Read for:* schema-based validation and skew detection at production scale. *Level:* intermediate.
- **Shreya Shankar, Rolando Garcia, Joseph M. Hellerstein, Aditya G. Parameswaran, "Operationalizing Machine Learning: An Interview Study"** (2022, arXiv:2209.09125). <https://arxiv.org/abs/2209.09125>
  *Read for:* how 18 ML engineers actually run production pipelines. *Level:* intro.
- **Andrei Paleyes, Raoul-Gabriel Urma, Neil D. Lawrence, "Challenges in Deploying Machine Learning: a Survey of Case Studies"** (2022, ACM Computing Surveys; arXiv:2011.09926). <https://arxiv.org/abs/2011.09926>
  *Read for:* reported deployment problems organized by workflow stage. *Level:* intro.
- **Stephan Rabanser, Stephan Günnemann, Zachary C. Lipton, "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"** (2019, NeurIPS; arXiv:1810.11953). <https://arxiv.org/abs/1810.11953>
  *Read for:* a careful comparison of shift detectors. *Level:* advanced.
- **Bilal Yurdakul, "Statistical Properties of Population Stability Index"** (2018, PhD dissertation, Western Michigan University). <https://scholarworks.wmich.edu/dissertations/3208/>
  *Read for:* PSI's sampling behavior and how its rules of thumb relate to error rates. *Level:* advanced.
- **Alec Warner and Štěpán Davidovič, with Alex Hidalgo, Betsy Beyer, Kyle Smith, Matt Duftler, "Canarying Releases"** (The Site Reliability Workbook, Google, free online). <https://sre.google/workbook/canarying-releases/>
  *Read for:* canary design: control groups, size, duration and metrics. *Level:* intro.
- **MLflow developers, "MLflow Model Registry"** (documentation, MLflow 3). <https://mlflow.org/docs/latest/ml/model-registry/>
  *Read for:* versions, aliases and tags in a widely used registry, and the deprecation of stages. *Level:* intro.
- **scikit-learn developers, "Model persistence"** (User Guide, version 1.9.1). <https://scikit-learn.org/stable/model_persistence.html>
  *Read for:* pickle's security and version risks, what metadata to save, and safer formats. *Level:* intro.
- **DVC developers, "Data versioning"** (DVC documentation). <https://doc.dvc.org/start/data-management/data-versioning>
  *Read for:* content-addressed data versioning with Git. *Level:* intro.

### L4-02: Efficient Training and Inference

- **sankarsh98 (GitHub), "AI Systems", Track B: Inference efficiency** (web course, free). <https://sankarsh98.github.io/ai-systems/inference/prefill-and-decode>
  *Read for:* prefill and decode, the KV cache, quantization and speculative decoding, with runnable NumPy code. *Level:* intro.
- **Hugging Face nanotron team, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters"** (2025, Hugging Face). <https://huggingface.co/spaces/nanotron/ultrascale-playbook>
  *Read for:* memory formulas, every kind of parallelism and ZeRO, explained with measurements from thousands of runs. *Level:* intro to intermediate.
- **PyTorch developers, "Automatic Mixed Precision package - torch.amp"** (PyTorch 2.15 documentation). <https://docs.pytorch.org/docs/2.15/amp.html>
  *Read for:* which operations autocast runs in 16 bits or fp32, and when `GradScaler` is needed. *Level:* intro.
- **Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (2019, arXiv:1910.02054). <https://arxiv.org/abs/1910.02054>
  *Read for:* the $16\Psi$ accounting and the memory and communication of each sharding stage. *Level:* intermediate.
- **Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro, "Reducing Activation Recomputation in Large Transformer Models"** (2022, arXiv:2205.05198). <https://arxiv.org/abs/2205.05198>
  *Read for:* the activation derivation of section 2.2, sequence parallelism and selective recomputation. *Level:* intermediate to advanced.
- **Paulius Micikevicius, Sharan Narang, Jonah Alben, et al. (11 authors), "Mixed Precision Training"** (2018, ICLR; arXiv:1710.03740). <https://arxiv.org/abs/1710.03740>
  *Read for:* master weights, loss scaling and fp32 accumulation. *Level:* intro to intermediate.
- **Paulius Micikevicius, Dusan Stosic, Neil Burgess, et al. (15 authors), "FP8 Formats for Deep Learning"** (2022, arXiv:2209.05433). <https://arxiv.org/abs/2209.05433>
  *Read for:* the E4M3 and E5M2 encodings and evidence that fp8 training can match 16-bit. *Level:* intermediate.
- **Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (2019, arXiv:1909.08053). <https://arxiv.org/abs/1909.08053>
  *Read for:* tensor parallelism's column and row splits, in a few lines of PyTorch. *Level:* intermediate.
- **Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al. (12 authors), "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (2021, SC21; arXiv:2104.04473). <https://arxiv.org/abs/2104.04473>
  *Read for:* combining tensor, pipeline and data parallelism, pipeline schedules and their bubbles, and the FLOP formula. *Level:* advanced.
- **Geoffrey Hinton, Oriol Vinyals, Jeff Dean, "Distilling the Knowledge in a Neural Network"** (2015, NIPS 2014 Deep Learning Workshop; arXiv:1503.02531). <https://arxiv.org/abs/1503.02531>
  *Read for:* temperature, soft targets and the $T^2$ factor. *Level:* intro.
- **Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag, "What is the State of Neural Network Pruning?"** (2020, MLSys; arXiv:2003.03033). <https://arxiv.org/abs/2003.03033>
  *Read for:* a meta-analysis of 81 pruning papers and a checklist for fair comparisons. *Level:* intermediate.
- **Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. (9 authors), "Efficient Memory Management for Large Language Model Serving with PagedAttention"** (2023, SOSP; arXiv:2309.06180). <https://arxiv.org/abs/2309.06180>
  *Read for:* why KV memory is wasted, and paging, sharing and preemption in vLLM. *Level:* intermediate.

### L4-03: Evaluation and Benchmarks

- **Evan Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"** (2024, arXiv:2411.00640). https://arxiv.org/abs/2411.00640
  *Read for:* the sampling model behind eval scores, clustered standard errors, paired differences, power analysis and resampling answers, with formulas you can use directly. *Level:* intro.
- **Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, Dan Jurafsky, "With Little Power Comes Great Responsibility"** (2020, EMNLP; arXiv:2010.06595). https://arxiv.org/abs/2010.06595
  *Read for:* how to do power analysis for NLP comparisons and human evaluations, and evidence that many published comparisons are underpowered. *Level:* intermediate.
- **Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing"** (2018, ACL). https://aclanthology.org/P18-1128/
  *Read for:* which significance tests fit which metrics, and how often published NLP work skipped testing. *Level:* intermediate.
- **Inioluwa Deborah Raji, Emily M. Bender, Amandalynne Paullada, Emily Denton, Alex Hanna, "AI and the Everything in the Whole Wide World Benchmark"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2111.15366). https://arxiv.org/abs/2111.15366
  *Read for:* the construct-validity critique of treating a few benchmarks as measures of general capability. *Level:* intro.
- **Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, et al. (30 authors), "Lessons from the Trenches on Reproducible Evaluation of Language Models"** (2024, arXiv:2405.14782). https://arxiv.org/abs/2405.14782
  *Read for:* how prompt format and implementation details change scores, from the maintainers of lm-evaluation-harness. *Level:* intermediate.
- **Curtis G. Northcutt, Anish Athalye, Jonas Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2103.14749). https://arxiv.org/abs/2103.14749
  *Read for:* how common test-label errors are and how they can reorder strong models. *Level:* intro.
- **Hugh Zhang et al. (15 authors), "A Careful Examination of Large Language Model Performance on Grade School Arithmetic"** (2024, NeurIPS Datasets and Benchmarks; arXiv:2405.00332). https://arxiv.org/abs/2405.00332
  *Read for:* building a fresh matched test set (GSM1k) to measure overfitting and contamination. Note that the abstract's headline number changed between versions (13% in v1, 8% in the camera-ready). *Level:* intermediate.
- **Ron Artstein, Massimo Poesio, "Inter-Coder Agreement for Computational Linguistics"** (2008, Computational Linguistics 34(4)). https://aclanthology.org/J08-4004/
  *Read for:* kappa, alpha and their relatives, their assumptions, and how to interpret them. Open access. *Level:* intermediate.
- **Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. (13 authors), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (2023, NeurIPS Datasets and Benchmarks; arXiv:2306.05685). https://arxiv.org/abs/2306.05685
  *Read for:* the case for LLM judges, their agreement with humans, and the first catalogue of their biases. *Level:* intro.
- **Peiyi Wang, Lei Li, Liang Chen, et al., "Large Language Models are not Fair Evaluators"** (2024, ACL; arXiv:2305.17926). https://aclanthology.org/2024.acl-long.511/
  *Read for:* position bias measured directly, and calibration methods that reduce it. The ACL version lists 11 authors, the arXiv version 10. *Level:* intermediate.
- **Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto, "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators"** (2024, COLM; arXiv:2404.04475). https://arxiv.org/abs/2404.04475
  *Read for:* the regression approach to removing length bias from judge win rates. *Level:* intermediate.
- **Shivalika Singh, Yiyang Nan, Alex Wang, et al. (13 authors), "The Leaderboard Illusion"** (2025, arXiv:2504.20879). https://arxiv.org/abs/2504.20879
  *Read for:* how private testing and data access asymmetries distort a crowd-sourced leaderboard; read with Arena's public response. *Level:* intermediate.

### L4-04: Interpretability

- **Cynthia Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead"** (2019, Nature Machine Intelligence 1:206-215; arXiv:1811.10154). https://arxiv.org/abs/1811.10154
  *Read for:* the case against post-hoc explanation in high-stakes settings. Open version on arXiv. *Level:* intro.
- **Yonatan Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances"** (2022, Computational Linguistics 48(1); arXiv:2102.12452). https://arxiv.org/abs/2102.12452
  *Read for:* a short, balanced review of what probing can and cannot show. *Level:* intro.
- **John Hewitt, Percy Liang, "Designing and Interpreting Probes with Control Tasks"** (2019, EMNLP; arXiv:1909.03368). https://aclanthology.org/D19-1275/
  *Read for:* control tasks and selectivity, the baseline every probing study needs. *Level:* intermediate.
- **Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, Been Kim, "Sanity Checks for Saliency Maps"** (2018, NeurIPS; arXiv:1810.03292). https://arxiv.org/abs/1810.03292
  *Read for:* the randomization tests and which popular saliency methods fail them. *Level:* intro.
- **Mukund Sundararajan, Ankur Taly, Qiqi Yan, "Axiomatic Attribution for Deep Networks"** (2017, ICML; arXiv:1703.01365). https://arxiv.org/abs/1703.01365
  *Read for:* integrated gradients and the axiomatic approach to attribution. *Level:* intermediate.
- **Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter, "Zoom In: An Introduction to Circuits"** (2020, Distill). https://distill.pub/2020/circuits/zoom-in/
  *Read for:* the features-and-circuits program and its three claims, with vision examples. *Level:* intro.
- **Nelson Elhage, Neel Nanda, Catherine Olsson, et al. (25 authors), "A Mathematical Framework for Transformer Circuits"** (2021, Transformer Circuits Thread). https://transformer-circuits.pub/2021/framework/index.html
  *Read for:* the residual-stream view, zero-, one- and two-layer attention-only models, and composition. *Level:* advanced.
- **Catherine Olsson, Nelson Elhage, Neel Nanda, et al. (26 authors), "In-context Learning and Induction Heads"** (2022, arXiv:2209.11895). https://arxiv.org/abs/2209.11895
  *Read for:* the phase change, the six lines of evidence, and how strong each is at different scales. *Level:* intermediate.
- **Nelson Elhage, Tristan Hume, Catherine Olsson, et al. (16 authors), "Toy Models of Superposition"** (2022, arXiv:2209.10652). https://transformer-circuits.pub/2022/toy_model/index.html
  *Read for:* when and how networks store more features than dimensions. *Level:* intermediate.
- **Trenton Bricken, Adly Templeton, Joshua Batson, et al. (24 authors), "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"** (2023, Transformer Circuits Thread), and **Adly Templeton, Tom Conerly, Jonathan Marcus, et al. (22 authors), "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"** (2024). https://transformer-circuits.pub/2023/monosemantic-features/index.html and https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
  *Read for:* sparse autoencoders from a one-layer model to a production model, with many worked features. *Level:* intermediate.
- **Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt, "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"** (2023, ICLR; arXiv:2211.00593). https://arxiv.org/abs/2211.00593
  *Read for:* a full circuit analysis in a real language model, with criteria for evaluating it. *Level:* advanced.
- **Lee Sharkey, Bilal Chughtai, et al. (29 authors), "Open Problems in Mechanistic Interpretability"** (2025, arXiv:2501.16496). https://arxiv.org/abs/2501.16496
  *Read for:* the field's own map of what is unsolved; read with the critiques in section 7, especially Kantamneni et al. ([arXiv:2502.16681](https://arxiv.org/abs/2502.16681)) and Paulo and Belrose ([arXiv:2501.16615](https://arxiv.org/abs/2501.16615)). *Level:* advanced.

### L4-05: Safety, Alignment and Governance

- **International AI Safety Report 2026, chaired by Yoshua Bengio, with Stephen Clare, Carina Prunkl, Maksym Andriushchenko, et al. (92 authors)** (published February 3, 2026; arXiv:2602.21012). https://internationalaisafetyreport.org/
  *Read for:* the broadest evidence review of capabilities, risks and mitigations, with explicit statements of uncertainty and disagreement. *Level:* intro to intermediate.
- **Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, Dan Mané, "Concrete Problems in AI Safety"** (2016, arXiv:1606.06565). https://arxiv.org/abs/1606.06565
  *Read for:* the ML-grounded framing of safety problems that much later work builds on. *Level:* intro.
- **Alexander Wei, Nika Haghtalab, Jacob Steinhardt, "Jailbroken: How Does LLM Safety Training Fail?"** (2023, arXiv:2307.02483). https://arxiv.org/abs/2307.02483
  *Read for:* the competing-objectives and mismatched-generalization explanation of jailbreaks. *Level:* intermediate.
- **Nicholas Carlini, Anish Athalye, Nicolas Papernot, Wieland Brendel, Jonas Rauber, Dimitris Tsipras, Ian Goodfellow, Aleksander Madry, Alexey Kurakin, "On Evaluating Adversarial Robustness"** (2019, arXiv:1902.06705). https://arxiv.org/abs/1902.06705
  *Read for:* how to evaluate a defense honestly; the methodology transfers to jailbreak and injection defenses. *Level:* intermediate.
- **Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, et al. (20 authors), "Alignment faking in large language models"** (2024, arXiv:2412.14093). https://arxiv.org/abs/2412.14093
  *Read for:* the most discussed demonstration of strategic behavior in training, with its setup and caveats. Pair it with Summerfield et al. (arXiv:2507.03409) for the critique. *Level:* intermediate.
- **Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr, "Membership Inference Attacks From First Principles"** (2022, IEEE Symposium on Security and Privacy; arXiv:2112.03570). https://arxiv.org/abs/2112.03570
  *Read for:* why TPR at low FPR is the right metric, and the LiRA attack. *Level:* intermediate.
- **Cynthia Dwork, Aaron Roth, "The Algorithmic Foundations of Differential Privacy"** (2014, Foundations and Trends in Theoretical Computer Science 9(3-4); free PDF from the author). https://www.cis.upenn.edu/~aaroth/privacybook.html
  *Read for:* the definitions, the Laplace and Gaussian mechanisms, and composition, from the source. *Level:* intermediate to advanced.
- **Alexandra Chouldechova, "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments"** (2017, Big Data 5(2); arXiv:1703.00056). https://arxiv.org/abs/1703.00056
  *Read for:* the identity in section 7 and its consequences, applied to COMPAS data. *Level:* intro to intermediate.
- **Solon Barocas, Moritz Hardt, Arvind Narayanan, "Fairness and Machine Learning: Limitations and Opportunities"** (MIT Press, 2023; free online). https://fairmlbook.org/
  *Read for:* the full treatment of fairness criteria, causality, and the limits of technical fixes. *Level:* intro to intermediate.
- **Richard Ren, Steven Basart, Adam Khoja, Alice Gatti, et al. (12 authors), "Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress?"** (2024, NeurIPS; arXiv:2407.21792). https://arxiv.org/abs/2407.21792
  *Read for:* a quantitative check of whether safety benchmarks measure something other than capability. *Level:* intermediate.
- **NIST, "Artificial Intelligence Risk Management Framework (AI RMF 1.0)"** (NIST AI 100-1, January 2023) and the Generative AI Profile (NIST AI 600-1, July 2024). https://www.nist.gov/itl/ai-risk-management-framework
  *Read for:* a widely used voluntary vocabulary and process for AI risk management. *Level:* intro.
- **European Commission, "AI Act" policy page** (accessed October 2026), with the legal text on EUR-Lex (Regulation (EU) 2024/1689, amended by Regulation (EU) 2026/1744) and article-by-article text on the AI Act Service Desk. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
  *Read for:* the current official timeline and structure of the AI Act; check here before relying on any date in this module. *Level:* intro.

### L4-06: Research Skills and Careers

- **S. Keshav, "How to Read a Paper"** (2007, ACM SIGCOMM Computer Communication Review 37(3)). Free copy: https://www.cs.princeton.edu/courses/archive/fall10/cos597B/papers/howtoread-keshav.pdf
  *Read for:* the three-pass method and the five Cs in two pages. *Level:* intro.
- **Richard W. Hamming, "You and Your Research"** (talk at Bellcore, March 7, 1986; transcript). https://www.cs.virginia.edu/~robins/YouAndYourResearch.html
  *Read for:* choosing important problems and working habits, from someone who did important work. *Level:* intro.
- **Zachary C. Lipton, Jacob Steinhardt, "Troubling Trends in Machine Learning Scholarship"** (2018, arXiv:1807.03341). https://arxiv.org/abs/1807.03341
  *Read for:* the four patterns to avoid in your own writing and to notice in others'. *Level:* intro.
- **Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d'Alché-Buc, Emily Fox, Hugo Larochelle, "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)"** (2021, JMLR 22(164)). https://jmlr.org/papers/v22/20-303.html
  *Read for:* what code policies, checklists and reproducibility challenges change in practice. *Level:* intro.
- **Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger, "Deep Reinforcement Learning that Matters"** (2018, AAAI; arXiv:1709.06560). https://arxiv.org/abs/1709.06560
  *Read for:* how seeds, hyperparameters and codebases change conclusions; the 10-seed experiment is the memorable one. *Level:* intermediate.
- **Gábor Melis, Chris Dyer, Phil Blunsom, "On the State of the Art of Evaluation in Neural Language Models"** (2017, arXiv:1707.05589; presented at ICLR 2018). https://arxiv.org/abs/1707.05589
  *Read for:* a tuned LSTM baseline beating newer architectures; the case for equal tuning effort. *Level:* intermediate.
- **Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, et al. (16 authors), "Accounting for Variance in Machine Learning Benchmarks"** (2021, MLSys; arXiv:2103.03098). https://arxiv.org/abs/2103.03098
  *Read for:* which sources of variance matter and how to compare methods given them. *Level:* intermediate.
- **Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice"** (2021, NeurIPS; arXiv:2108.13264). https://arxiv.org/abs/2108.13264
  *Read for:* interval estimates, the interquartile mean and performance profiles for reporting few-run results. *Level:* intermediate.
- **NeurIPS, "NeurIPS Paper Checklist Guidelines"** (official page, 2026 version, accessed October 2026). https://neurips.cc/public/guides/PaperChecklist ; and the **ML Reproducibility Challenge** (official site). https://reproml.org/
  *Read for:* what reviewers now expect a paper to report, and a structured way to publish a reproduction. *Level:* intro.
- **Andrej Karpathy, "A Recipe for Training Neural Networks"** (blog, April 25, 2019). https://karpathy.github.io/2019/04/25/recipe/
  *Read for:* a practical sequence (data first, dumb baselines, overfit, regularize, tune) that prevents most wasted experiments. *Level:* intro.
- **PyTorch documentation, "Reproducibility"** (version 2.15). https://docs.pytorch.org/docs/2.15/notes/randomness.html ; **MLflow Tracking** documentation. https://mlflow.org/docs/latest/ml/tracking/
  *Read for:* what determinism PyTorch does and does not promise, and a production version of `tracker.py`. *Level:* intro.
- **Open Source Guides, "How to Contribute to Open Source"** (GitHub's opensource.guide). https://opensource.guide/how-to-contribute/ ; **80,000 Hours, "AI safety technical research" career review** (Benjamin Hilton, updated October 2026). https://80000hours.org/career-reviews/ai-safety-researcher/
  *Read for:* the mechanics and etiquette of a first contribution; one organization's view of the path into safety research. *Level:* intro.
