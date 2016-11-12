% This is the code to implement the algorithm 1 in paper
%"Twin Learning for Similarity  and Clustering: A Unified Kernel Approach,
%Zhao Kang; Chong Peng; Qiang Cheng, AAAI-17." 
%Please contact Zhao Kang if you have any question.
function [result]=unifiedcluster(K,s,alpha,beta)
% s is the true class label.
[m,n]=size(K);
Z=eye(n);
c=length(unique(s));
%options = optimset( 'Algorithm','interior-point-convex','Display','off');
for i=1:200
    Zold=Z;
    Z= (Z+Z')/2;
    D = diag(sum(Z));
    L = D-Z;
    
    [F, temp, ev]=eig1(L, c, 0);
    
    
    for ij=1:n
        for ji=1:n
            all(ji)=norm(F(ij,:)-F(ji,:));
        end
        
        H=2*alpha*eye(n)+2*K;
        H=(H+H')/2;
        ff=beta/2*all'-2*K(:,ij);
% we use the free package to solve quadratic equation: http://sigpromu.org/quadprog/index.html
        [Z(:,ij),err,lm] = qpas(H,ff,[],[],ones(1,n),1,zeros(n,1),ones(n,1));
% Z(:,ij)=quadprog(H,(beta/2*all'-2*K(:,ij))',[],[],ones(1,n),1,zeros(n,1),ones(n,1),Z(:,ij),options);
    end
    if i>5 &((norm(Z-Zold)/norm(Zold))<1e-3)
        break
    end
    
end
actual_ids= kmeans(F, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
[result] = ClusteringMeasure( actual_ids,s);
