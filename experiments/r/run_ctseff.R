# Usage: Rscript run_ctseff.R in.csv out.csv a_lo a_hi n_pts [drop_interactions]
# The curve is estimated over the full observed dose range (its bandwidth risk needs every
# observed dose); a_lo/a_hi are kept for the caller's interpolation.
suppressMessages({library(npcausal); library(SuperLearner)})
args <- commandArgs(TRUE)
d <- read.csv(args[1]); n.pts <- as.integer(args[5])
y <- d$Y; a <- d$A; x <- as.matrix(d[, grepl("^x", names(d))])
lib <- c("SL.earth", "SL.gam", "SL.glm", "SL.glm.interaction", "SL.mean", "SL.ranger")
# With many covariates, all pairwise interactions do not fit in memory; drop that learner.
if (length(args) >= 6 && args[6] == "1") lib <- setdiff(lib, "SL.glm.interaction")
bw <- seq(0.05, 1.5, length.out = 30) * sd(a)
r <- ctseff(y, a, x, bw.seq = bw, n.pts = n.pts, sl.lib = lib)
write.csv(r$res, args[2], row.names = FALSE)
