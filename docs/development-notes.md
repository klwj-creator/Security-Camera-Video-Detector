# Development notes

## 14 November 2026: reference-tag matching

The initial rotation bank padded every tag template to a mean-colored square. That unnecessarily penalized the low-contrast pattern on light clothing. Right-angle templates now retain their exact rectangular support; a clear light-shirt example reaches approximately 0.95 normalized correlation.

An initial orientation test incorrectly assumed that an upside-down repeated-bar pattern had a unique matching angle. On a uniform background, a small translation can make another orientation match equally well. The test now checks localization and detection strength rather than an unjustified orientation claim. Treat the exported angle as a template-bank hypothesis, not a reliable physical orientation measurement.

The project currently has no established Git repository/lesson-file convention. These implementation findings are recorded here alongside the source documents.
