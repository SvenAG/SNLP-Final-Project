require(plyr)
require(ggplot2)


rocdata <- function(grp, pred){
  # Produces x and y co-ordinates for ROC curve plot
  # Arguments: grp - labels classifying subject status
  #            pred - values of each observation
  # Output: List with 2 components:
  #         roc = data.frame with x and y co-ordinates of plot
  #         stats = data.frame containing: area under ROC curve, p value, upper and lower 95% confidence interval
  
  grp <- as.factor(grp)
  if (length(pred) != length(grp)) {
    stop("The number of classifiers must match the number of data points")
  } 
  
  if (length(levels(grp)) != 2) {
    stop("There must only be 2 values for the classifier")
  }
  
  cut <- unique(pred)
  tp <- sapply(cut, function(x) length(which(pred > x & grp == levels(grp)[2])))
  fn <- sapply(cut, function(x) length(which(pred < x & grp == levels(grp)[2])))
  fp <- sapply(cut, function(x) length(which(pred > x & grp == levels(grp)[1])))
  tn <- sapply(cut, function(x) length(which(pred < x & grp == levels(grp)[1])))
  tpr <- tp / (tp + fn)
  fpr <- fp / (fp + tn)
  roc = data.frame(x = fpr, y = tpr)
  roc <- roc[order(roc$x, roc$y),]
  
  i <- 2:nrow(roc)
  auc <- (roc$x[i] - roc$x[i - 1]) %*% (roc$y[i] + roc$y[i - 1])/2
  
  pos <- pred[grp == levels(grp)[2]]
  neg <- pred[grp == levels(grp)[1]]
  q1 <- auc/(2-auc)
  q2 <- (2*auc^2)/(1+auc)
  se.auc <- sqrt(((auc * (1 - auc)) + ((length(pos) -1)*(q1 - auc^2)) + ((length(neg) -1)*(q2 - auc^2)))/(length(pos)*length(neg)))
  ci.upper <- auc + (se.auc * 0.96)
  ci.lower <- auc - (se.auc * 0.96)
  
  se.auc.null <- sqrt((1 + length(pos) + length(neg))/(12*length(pos)*length(neg)))
  z <- (auc - 0.5)/se.auc.null
  p <- 2*pnorm(-abs(z))
  
  stats <- data.frame (auc = auc,
                       p.value = p,
                       ci.upper = ci.upper,
                       ci.lower = ci.lower
  )
  
  return (list(roc = roc, stats = stats))
}

rocplot.single <- function(grp, pred, title = "ROC Plot", p.value = FALSE){
  require(ggplot2)
  plotdata <- rocdata(grp, pred)
  
  if (p.value == TRUE){
    annotation <- with(plotdata$stats, paste("AUC = ",signif(auc, 2), sep=""))
  } else {
    annotation <- with(plotdata$stats, paste("AUC = ",signif(auc, 2), sep=""))
  }
  
  p <- ggplot(plotdata$roc, aes(x = x, y = y)) +
    geom_line(aes(colour = "")) +
    geom_abline(colour="azure3", intercept = 0, slope = 1) +
    theme_bw() +
    scale_x_continuous("False Positive Rate") +
    scale_y_continuous("True Positive Rate") +
    scale_colour_manual(labels = annotation, values = "#000000") +
    opts(title = title,
         plot.title = theme_text(face="bold", size=14), 
         axis.title.x = theme_text(face="bold", size=12),
         axis.title.y = theme_text(face="bold", size=12, angle=90),
         panel.grid.major = theme_blank(),
         panel.grid.minor = theme_blank(),
         legend.justification=c(1,0), 
         legend.position=c(1,0),
         legend.title=theme_blank(),
         legend.key = theme_blank()
    )
  return(p)
}

rocplot.multiple <- function(test.data.list, groupName = "label", predName = "probability", title = "ROC Plot", p.value = TRUE){
  plotdata <- llply(test.data.list, function(x) with(x, rocdata(grp = eval(parse(text = groupName)), pred = eval(parse(text = predName)))))
  plotdata <- list(roc = ldply(plotdata, function(x) x$roc),
                   stats = ldply(plotdata, function(x) x$stats)
  )
  
  if (p.value == TRUE){
    annotation <- with(plotdata$stats, paste("AUC = ",signif(auc, 2), sep=""))
  } else {
    annotation <- with(plotdata$stats, paste("AUC = ",signif(auc, 2), sep=""))
  }
  
  p <- ggplot(plotdata$roc, aes(x = x, y = y)) +
    geom_line(aes(colour = .id)) +
    geom_abline(colour = "azure3", intercept = 0, slope = 1) +
    theme_bw() +
    scale_x_continuous("False Positive Rate") +
    scale_y_continuous("True Positive Rate") +
    scale_colour_brewer(palette="Set1", breaks = names(test.data.list), labels = paste(names(test.data.list), ": ", annotation, sep = "")) +
    opts(title = title,
         plot.title = theme_text(size=14), 
         axis.title.x = theme_text(size=16),
         axis.title.y = theme_text(size=16, angle=90),
         panel.grid.major = theme_blank(),
         panel.grid.minor = theme_blank(),
         legend.justification=c(1,0), 
         legend.position=c(1,0),
         legend.text=theme_text(size=14),
         legend.title=theme_blank(),
         legend.key = theme_blank()
    )
  return(p)
}

d_1 <- as.data.frame(read.csv('/Users/robert/Dropbox/Stern/Year\ 2/Semester\ 1/CSCI-GA\ 3033\ SNLP/Project/codebase/data/test'))
d_2 <- as.data.frame(read.csv('/Users/robert/Dropbox/Stern/Year\ 2/Semester\ 1/CSCI-GA\ 3033\ SNLP/Project/codebase/data/test2'))

data <- list(Uno = d_1, Dos = d_2)
rocplot.multiple(data, title="", p.value = FALSE)