# Cumulative link mixed models for the human study (plan T7.1).
# Rscript examples/act_three/experiments/analysis/human_clmm.R
suppressPackageStartupMessages({library(ordinal); library(emmeans); library(jsonlite)})

root <- "examples/act_three/experiments/results/analysis"
d <- read.csv(file.path(root, "human_long.csv"), stringsAsFactors = FALSE)
d$cond <- factor(d$cond, levels = c("unsteered", "pe_only", "repe_only", "hybrid"))
d$pid <- factor(d$pid); d$vig <- factor(d$vig)
d$rating <- ordered(d$rating, levels = 1:7)

out <- list()
for (item in c("appropriate", "natural", "non_sycophantic")) {
  di <- d[d$item == item, ]
  m0 <- clmm(rating ~ 1 + (1 | pid) + (1 | vig), data = di, link = "logit", Hess = TRUE)
  m1 <- clmm(rating ~ cond + (1 | pid) + (1 | vig), data = di, link = "logit", Hess = TRUE)
  lr <- anova(m0, m1)
  res <- list(model = "clmm: rating ~ cond + (1|pid) + (1|vig)", n = nrow(di),
              AIC_m0 = AIC(m0), AIC_m1 = AIC(m1),
              LR_chisq = lr[2, "LR.stat"], LR_df = lr[2, "Df"], LR_p = lr[2, "Pr(>Chisq)"])
  cf <- summary(m1)$coefficients
  cn <- grep("^cond", rownames(cf), value = TRUE)
  res$coef <- lapply(cn, function(n) list(term = n, est = cf[n, 1], se = cf[n, 2], z = cf[n, 3], p = cf[n, 4]))
  names(res$coef) <- cn
  vc <- VarCorr(m1)
  res$random_var <- list(pid = as.numeric(vc$pid[1]), vig = as.numeric(vc$vig[1]))
  # random condition slopes per participant (plan: keep if converges and improves AIC)
  m2 <- tryCatch(clmm(rating ~ cond + (1 + cond | pid) + (1 | vig), data = di, link = "logit"),
                 error = function(e) NULL, warning = function(w) NULL)
  res$slopes_model <- if (is.null(m2)) list(converged = FALSE) else list(converged = TRUE, AIC = AIC(m2), better = AIC(m2) < AIC(m1))
  res$model_used <- if (!is.null(m2) && AIC(m2) < AIC(m1)) "random slopes" else "random intercepts"
  mf <- if (!is.null(m2) && AIC(m2) < AIC(m1)) m2 else m1   # model used for the contrasts (AIC rule)
  res$contrast_model <- if (identical(mf, m1)) "random intercepts (pid, vig)" else "random participant condition slopes + vig intercept"
  if (!identical(mf, m1)) {
    cf2 <- summary(m2)$coefficients; cn2 <- grep("^cond", rownames(cf2), value = TRUE)
    res$coef_slopes_model <- lapply(cn2, function(n) list(term = n, est = cf2[n, 1], se = cf2[n, 2], z = cf2[n, 3], p = cf2[n, 4]))
  }
  em <- emmeans(mf, ~ cond)
  res$emm <- as.data.frame(em)
  ct <- as.data.frame(pairs(em, adjust = "holm"))
  res$contrasts_holm <- ct
  # all pairwise contrasts with CIs on the log-odds scale
  res$contrasts_ci <- as.data.frame(confint(pairs(em, adjust = "holm")))
  out[[item]] <- res
}
write_json(out, file.path(root, "human_clmm.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("wrote human_clmm.json\n")
