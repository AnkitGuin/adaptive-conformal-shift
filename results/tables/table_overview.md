### Overview – coverage / mean interval score for every scenario (mean over seeds)

| Scenario | Static split | Rolling window | Exp. weighted | ACI | DA-ACP | CQR (static) | CQR + DA-ACP |
|---|---|---|---|---|---|---|---|
| A: IID control | 0.900 / 209 | 0.899 / 207 | 0.913 / 205 | 0.900 / 219 | 0.908 / 204 | 0.900 / 180 | 0.908 / 180 |
| A': Natural (chronological) | 0.841 / 498 | 0.905 / 465 | 0.915 / 465 | 0.900 / 471 | 0.910 / 465 | 0.849 / 554 | 0.920 / 513 |
| B: Gradual covariate | 0.844 / 510 | 0.897 / 477 | 0.909 / 477 | 0.900 / 478 | 0.905 / 480 | 0.872 / 560 | 0.917 / 506 |
| C: Abrupt covariate | 0.810 / 627 | 0.896 / 537 | 0.909 / 538 | 0.900 / 545 | 0.906 / 542 | 0.857 / 610 | 0.916 / 537 |
| C': Covariate resampling | 0.777 / 583 | 0.902 / 498 | 0.914 / 493 | 0.902 / 508 | 0.909 / 495 | 0.785 / 685 | 0.923 / 599 |
| D: Noise increase | 0.802 / 574 | 0.904 / 516 | 0.913 / 516 | 0.900 / 529 | 0.908 / 515 | 0.808 / 626 | 0.915 / 569 |
| E: Conditional shift | 0.682 / 1231 | 0.899 / 853 | 0.916 / 847 | 0.901 / 830 | 0.915 / 851 | 0.683 / 1291 | 0.919 / 921 |
