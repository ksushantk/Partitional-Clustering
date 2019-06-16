%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes: This demo program contains the basic Flower Pollination       %
% algorithm for clustering problem without fine-tuning the parameters. %
% Though this demo works very well, it is expected that this           %
% illustration with randomly generated synthetic dataset is much less  %
% efficient than the work reported in the paper.                       % 
%                                                                      %
% Citation details:                                                    %
% 1)Xin-She Yang, Flower pollination algorithm for global optimization,%
% Unconventional Computation and Natural Computation,                  %
% Lecture Notes in Computer Science, Vol. 7445, pp. 240-249 (2012).    %
% 2)X. S. Yang, M. Karamanoglu, X. S. He, Multi-objective flower       %
% algorithm for optimization, Procedia in Computer Science,            %
% vol. 18, pp. 861-868 (2013).                                         %
% 3) J. Senthilnath, Sushant Kulkarni, S. Suresh, X.S. Yang,           %
%  J.A. Benediktsson,(2019) "FPA Clust: Evaluation of Flower           %
%  Pollination Algorithm for Data Clustering", Evolutionary            % 
%  Intelligence, DOI: 10.1007/s12065-019-00254-1                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [best] = Flower_Pollination(traindat, limits, attr)
% Iteration parameters
N_iter = 100;            % Total number of iterations
n = 5;                  % population size
p = 0.8;
% Dimension of the search variables
d = attr;
Lb = limits(2,:);
Ub = limits(1,:);
tem = zeros(2, d);
% Initialize the population/solutions
for i=1:n,
    Sol(i,:)=Lb+(Ub-Lb).*rand(1,d);
    Fitness(i)=Fun(Sol(i,:));
end
% plot initial solutions for visualization
plot(Sol(:,1), Sol(:,2),'gs', 'LineWidth',1.5);
hold on;
% Find the current best solution
[fmin, I] = min(Fitness);
best = Sol(I,:);
S = Sol;
% Start the iterations -- Flower Algorithm
for t=1:N_iter
    % Loop over all solutions
    for i=1:n
        % Pollens are carried by insects and thus can move in
        % large scale, large distance.
        % This L should replace by Levy flights
        % Formula: x_i^{t+1}=x_i^t+ L (x_i^t-gbest)
        if rand > p
            %% L=rand;
            L = Levy(d);
            dS = L.*(Sol(i,:) - best);
            S(i,:) = Sol(i,:)+dS;
            tem(1,:) = Sol(i,:);               % Solution before movement
            % Check if the simple limits/bounds are OK
            S(i,:)=simplebounds(S(i,:),Lb,Ub);
            
            % If not, then local pollenation of neighbor flowers
        else
            epsilon=rand;
            % Find random flowers in the neighbourhood
            JK=randperm(n);
            % As they are random, the first two entries also random
            % If the flower are the same or similar species, then
            % they can be pollenated, otherwise, no action.
            % Formula: x_i^{t+1}+epsilon*(x_j^t-x_k^t)
            S(i,:)=S(i,:)+epsilon*(Sol(JK(1),:)-Sol(JK(2),:));
            tem(1,:) = Sol(i,:);               % Solution before movement
            % Check if the simple limits/bounds are OK
            S(i,:)=simplebounds(S(i,:),Lb,Ub);
        end
        
        % Evaluate new solutions
        Fnew=Fun(S(i,:));
        % If fitness improves (better solutions found), update then
        if (Fnew<=Fitness(i)),
            Sol(i,:)=S(i,:);                 % Replace initial solution with improvised solution
            tem(2,:) = S(i,:);               % Solution after movement
            Fitness(i)=Fnew;                 % Replace initial fitness with improvised fitness
        end
        
        % Update the current global best
        if Fnew<=fmin,
            best=S(i,:);
            fmin=Fnew;
        end
        if all(tem(2,:) == 0,2) == 1
            tem(2,:) = tem(1,:);
        end
        plot(tem(:,1),tem(:,2),'k:');
    end
end
% plot the final optimal cluster center
plot(best(1),best(2),'k*', 'LineWidth',3)
legend('Class 1 Training','Class 2 Training','Class 1 Testing','Class 2 Testing ','Agents','Agents Movement','Location','NorthEastOutside')
% legend('Agents','Agents Movement','Location','NorthEastOutside')
xlabel('X'); ylabel('Y'); xlim([0 25]); ylim([0 25]); grid on
text(27,17,'* Cluster Centers', 'FontName','Times','Fontsize',12)
% Output/display
% disp(['Total number of evaluations: ', num2str(N_iter*n)]);
% disp(['Cluster center=',num2str(best)]);
% disp(['Optimal value=',num2str(fmin)]);
% Application of simple constraints
    function s=simplebounds(s,Lb,Ub)
        % Apply the lower bound
        ns_tmp=s;
        I=ns_tmp<Lb;
        ns_tmp(I)=Lb(I);
        
        % Apply the upper bounds
        J=ns_tmp>Ub;
        ns_tmp(J)=Ub(J);
        % Update this new move
        s=ns_tmp;
    end
% Draw n Levy flight sample
    function L = Levy(d)
        % Levy exponent and coefficient
        % For details, see Chapter 11 of the following book:
        % Xin-She Yang, Nature-Inspired Optimization Algorithms, Elsevier, (2014).
        beta=3/2;
        sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
        u=randn(1,d)*sigma;
        v=randn(1,d);
        step=u./abs(v).^(1/beta);
        L=0.01*step;
    end
    function z = Fun(u)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OBJECTIVE FUNCTION - Evaluation of fitness value of each solution vector
        % x receives the matrix containing the pseudo optimal cluster centres to be evaluated
        % Optimisation of Clustering Criteria by Reformulation
        % Refer Equation (4) in paper: "Optimisation of Clustering Criteria by Reformulation
        % Note: your own objective function can be written here
        % when using your own function, remember to change limits/bounds
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        z = sum(pdist2(u,traindat));
    end
end