%%-------------------------------------------------------------------------
% (Citation details):                                                    % 
% J. Senthilnath, Sushant Kulkarni, J.A. Benediktsson and X.S. Yang      %
% (2016) "A novel approach for Multi-Spectral Satellite Image            %
% Classification based on the Bat Algorithm clustering problems",        %
% IEEE Letters for Geoscience and Remote Sensing Letters,                %
%  Vol. 13, No. 4, pp.599�603.                                           %
%%-------------------------------------------------------------------------
function [best]=Bat_Algorithm(traindat,limits,v)
% Default parameters
n= 5;                                 % Population size, typically 10 to 40
N_gen= 50;                       % Number of generations
% Iteration parameters
A=zeros(n,1);                    % Loudness  (constant or decreasing)
r= zeros(n,1);                    % Pulse rate (constant or decreasing)
% The frequency range determines the scalings
% These values need to be changed as necessary
Qmin=0;                           % Frequency minimum
Qmax=2;                          % Frequency maximum
% Dimension of the search variables
d=v;                                  % Number of dimensions 
N_iter=0;                           % Total number of function evaluations
% Upper limit/bounds/ a vector
Ub=limits(1,:);
% Lower limit/bounds/ a vector
Lb=limits(2,:);
% Initializing arrays
Q=zeros(n,1);                     % Frequency of Bats
v=zeros(n,d);                      % Velocities of Bats
% Initialize the population/solutions
for i=1:n,
  Sol(i,:)=Lb+(Ub-Lb).*rand(1,d);
  Fitness(i)=Fun(Sol(i,:));
   r(i)=rand(1);
   A(i)=1+rand(1);
end
r0=r;
plot(Sol(:,1),Sol(:,2),'gs', 'LineWidth',1.5);     % plot initial solutions for visualization
hold on;
% Find the initial best solution
% Here, probable center with least distance in cluster
[fmin, I]=min(Fitness);
best=Sol(I,:);
% Start of iterations -- Bat Algorithm (essential part)  %
for t=1:N_gen
        % Loop over all bats/solutions
        for i=1:n
            Q(i)=Qmin+(Qmax-Qmin)*rand;
            v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
            S(i,:)=Sol(i,:)+v(i,:);
            tem(1,:) = Sol(i,:);                  % Solution before movement
            % Apply simple bounds/limits
            S(i,:)=simplebounds(S(i,:),Lb,Ub);
            % Pulse rate
            if rand>r(i)
                % The factor 0.001 limits the step sizes of random walks 
                S(i,:)=S(i,:)+0.001*randn(1,d);
                S(i,:)=simplebounds(S(i,:),Lb,Ub);
            end
           % Evaluate new solutions
           Fnew=Fun(S(i,:));
           % Update if the solution improves, or not too loud
           if (Fnew<=Fitness(i)) && (rand<A(i)) 
                Sol(i,:)=S(i,:);                   % Replace initial solution with improvised solution
                tem(2,:) = S(i,:);                % Solution after movement
                Fitness(i)=Fnew;              % Replace initial fitness with improvised fitness
                A(i)=0.9*A(i);                  % Update the Loudness of Bats
                r(i)=r0(i)*(1-exp(-0.9*N_gen)); % Update the Pitch of Bats
           end
          % Find and update the current best solution
          if Fnew<=fmin,
                best=S(i,:);
                fmin=Fnew;
          end
        
        % plot the movement of the solutions
        pause(0.005)
        hold on;
        plot(tem(:,1),tem(:,2),'k:');  
      
        end
        N_iter=N_iter+n;
end
% plot the final optimal cluster center
plot(best(1),best(2),'k*', 'LineWidth',3) 
legend('Class 1 Training','Class 2 Training','Class 1 Testing','Class 2 Testing ','Agents','Agents Movement','Location','NorthEastOutside')
text(33,23,'* Cluster Centers', 'FontName','Times','Fontsize',12)
 
% Output/display
disp(['Number of evaluations: ',num2str(N_iter)]);
disp(['Best =',num2str(best),' fmin=',num2str(fmin)]);
% Application of simple limits/bounds
    function s=simplebounds(s,Lb,Ub)
        % Apply the lower bound vector
        ns_tmp=s;
        tt=ns_tmp<Lb;
        ns_tmp(tt)=Lb(tt);
  
        % Apply the upper bound vector 
        J=ns_tmp>Ub;
        ns_tmp(J)=Ub(J);
        % Update this new move 
        s=ns_tmp;
        
     end
function objective = Fun(x)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % OBJECTIVE FUNCTION - Evaluation of fitness value of each solution vector
    % x receives the matrix containing the pseudo optimal cluster centres to be evaluated
    % Optimisation of Clustering Criteria by Reformulation
    % Refer Equation (4) in paper: "Optimisation of Clustering Criteria by Reformulation
    % Note: your own objective function can be written here
    % when using your own function, remember to change limits/bounds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    objective = sum(pdist2(x,traindat)); 
end
end