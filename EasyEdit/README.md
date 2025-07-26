# EasyEdit fork from BabelEdits

This is a fork from [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) which
implements the following changes necessary to the Babeledits code:

- It adds the BabelReFT method as well as other baselines like ReFT
- It implements a different evaluation strategy as EasyEdit (see Appendix A.1 of the [BabelEdits paper](https://aclanthology.org/2025.findings-acl.438.pdf))

We chose to merge the fork history with our babeledits repo to keep all the code
in one place and to make it more easily runnable. Nevertheless, the git history
of EasyEdit was kept in the merge so as to preserve the initial contribution of
the EasyEdit authors.