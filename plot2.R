align_legend <- function(p, hjust = 0.5)
{
  # extract legend
  g <- cowplot::plot_to_gtable(p)
  grobs <- g$grobs
  legend_index <- which(sapply(grobs, function(x) x$name) == "guide-box")
  legend <- grobs[[legend_index]]
  
  # extract guides table
  guides_index <- which(sapply(legend$grobs, function(x) x$name) == "layout")
  
  # there can be multiple guides within one legend box  
  for (gi in guides_index) {
    guides <- legend$grobs[[gi]]
    
    # add extra column for spacing
    # guides$width[5] is the extra spacing from the end of the legend text
    # to the end of the legend title. If we instead distribute it by `hjust:(1-hjust)` on
    # both sides, we get an aligned legend
    spacing <- guides$width[5]
    guides <- gtable::gtable_add_cols(guides, hjust*spacing, 1)
    guides$widths[6] <- (1-hjust)*spacing
    title_index <- guides$layout$name == "title"
    guides$layout$l[title_index] <- 2
    
    # reconstruct guides and write back
    legend$grobs[[gi]] <- guides
  }
  
  # reconstruct legend and write back
  g$grobs[[legend_index]] <- legend
  g
}

d<-read.csv('data/correct.csv')
d$X<-NULL
dp<-data.frame(probs=as.vector(t(d)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))

library(ggplot2)
library(viridisLite)
library(patchwork)
library(ggridges)

p1<-ggplot(data.frame(dp), aes(y = lrate, x = alpha, fill=probs)) +
  geom_tile()+
  scale_fill_viridis_c(limits = c(0,1), breaks = c(0, 0.2,0.4,0.6,0.8, 1)) + 
  guides(breaks =seq(-0.2, 1, 0.2), fill = guide_colourbar(barwidth = 1, title="    p(valid)     "))+
  theme_classic() +
  scale_y_continuous(expand=c(0,0))+
  #define new breaks on x-axis
  scale_x_continuous(expand=c(0,0))+
  xlab(expression('Inverse temperature'~beta))+ 
  ylab(expression('Learning rate'~alpha))+
  ggtitle("a) Performance")+
  #adjust text size
  theme(text = element_text(size=18,  family="sans"))
p1

d1<-read.csv('data/esolutions.csv')
d2<-read.csv('data/dsolutions.csv')
d3<-read.csv('data/esolutionscontrol.csv')
d4<-read.csv('data/dsolutionscontrol.csv')

d1$X<-d2$X<-d3$X<-d4$X<- NULL
dp1<-data.frame(probs=as.vector(t(d1)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))
dp2<-data.frame(probs=as.vector(t(d2)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))
dp3<-data.frame(probs=as.vector(t(d3)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))
dp4<-data.frame(probs=as.vector(t(d4)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))
dp<-rbind(dp1, dp2, dp3, dp4)
dp$solution<-rep(rep(c("E-Solutions", "D-Solutions"), each=nrow(dp1)), 2)
dp$condition<-rep(c('Experimental Group', 'Control Group'), each=nrow(dp1)*2)
dp$solution<-factor(dp$solution, levels=c("E-Solutions", "D-Solutions"))
dp$condition<-factor(dp$condition, c('Experimental Group', 'Control Group'))

p2<-ggplot(dp, aes(y = lrate, x = alpha, fill=probs)) +
  geom_tile()+
  scale_fill_viridis_c(limits = c(0,1), breaks = c(0, 0.2,0.4,0.6,0.8, 1)) + 
  guides(breaks =seq(-0.2, 1, 0.2), fill = guide_colourbar(barwidth = 1, barheight = 20, title="p"))+
  theme_classic() +
  scale_y_continuous(expand=c(0,0))+
  #define new breaks on x-axis
  scale_x_continuous(expand=c(0,0))+
  xlab(expression('Inverse temperature'~beta))+ 
  ylab(expression('Learning rate'~alpha))+
  ggtitle("c) Einstellung Effect")+
  facet_grid(solution~condition)+
  #adjust text size
  theme(text = element_text(size=18,  family="sans"))
p2

d$alpha<-factor(rep(c("b=1","b=100" ), each=nrow(d1)),
                labels=c("beta==1", "beta==100"))

d<-read.csv('data/klds.csv')
d$X<-NULL
dp<-data.frame(probs=as.vector(t(d)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))

p4<-ggplot(data.frame(dp), aes(y = lrate, x = alpha, fill=probs)) +
  geom_tile()+
  scale_fill_viridis_c(limits = c(0,10), breaks = c(0, 2,4,6,8, 10)) + 
  guides(breaks =seq(-0.2, 1, 0.2), fill = guide_colourbar(barwidth = 1,  title="log(samples)"))+
  theme_classic() +
  scale_y_continuous(expand=c(0,0))+
  #define new breaks on x-axis
  scale_x_continuous(expand=c(0,0))+
  xlab(expression('Inverse temperature'~beta))+ 
  ylab(expression('Learning rate'~alpha))+
  ggtitle("b) Decision Time")+
  #adjust text size
  theme(text = element_text(size=18,  family="sans"))
p4


pdf('figures/einstellung.pdf', width=11, height=6)
((p1 / p4) | p2)+plot_layout(widths = c(1,2))
dev.off()