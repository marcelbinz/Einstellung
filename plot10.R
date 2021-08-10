d<-read.csv('data/esolutions.csv')
d$X<-NULL
dp<-data.frame(probs=as.vector(t(d)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))

library(ggplot2)
library(viridisLite)
library(patchwork)
library(ggridges)

p1<-ggplot(data.frame(dp), aes(y = lrate, x = alpha, fill=probs)) +
  geom_tile()+
  scale_fill_viridis_c(limits = c(0,1), breaks = c(0, 0.2,0.4,0.6,0.8, 1)) + 
  guides(breaks =seq(-0.2, 1, 0.2), fill = guide_colourbar(barwidth = 1, barheight = 14.5, title="p"))+
  theme_classic() +
  scale_y_continuous(expand=c(0,0))+
  #define new breaks on x-axis
  scale_x_continuous(expand=c(0,0))+
  xlab(expression('Inverse temperature'~beta))+ 
  ylab(expression('Learning rate'~alpha))+
  ggtitle("a) E-solutions (full model)")+
  #adjust text size
  theme(text = element_text(size=18,  family="sans"))
p1

d<-read.csv('data/esolutions_no_physical.csv')
d$X<-NULL
dp<-data.frame(probs=as.vector(t(d)), lrate=rep(seq(0, 1, 0.05), 50), alpha=rep(1:50, each=21))

library(ggplot2)
library(viridisLite)
library(patchwork)
library(ggridges)

p2<-ggplot(data.frame(dp), aes(y = lrate, x = alpha, fill=probs)) +
  geom_tile()+
  scale_fill_viridis_c(limits = c(0,1), breaks = c(0, 0.2,0.4,0.6,0.8, 1)) + 
  guides(breaks =seq(-0.2, 1, 0.2), fill = guide_colourbar(barwidth = 1, barheight = 14.5, title="p"))+
  theme_classic() +
  scale_y_continuous(expand=c(0,0))+
  #define new breaks on x-axis
  scale_x_continuous(expand=c(0,0))+
  xlab(expression('Inverse temperature'~beta))+ 
  ylab(expression('Learning rate'~alpha))+
  ggtitle("b) E-solutions (ablated model)")+
  #adjust text size
  theme(text = element_text(size=18,  family="sans"))
p2

pdf('figures/nophysical.pdf', width=11, height=4.5)
(p1 | p2)+plot_layout(widths = c(1,1))
dev.off()