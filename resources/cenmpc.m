classdef cenmpc
% CENMPC design a centralized MPC controller. Robustness is obtained designing a tube-based MPC controller
% 
% ctrl = cenmpc (objlss,N,flag,options)
%
% ---------------------------------------------------------------------------
% DESCRIPTION
% ---------------------------------------------------------------------------  
% Return a centralized MPC controller without setting the terminal penalty 
% and terminal region
% 
% ctrl = CENMPC (objlss,N)
% ctrl = CENMPC (objlss,N,flag) 
% ctrl = CENMPC (objlss,N,flag,options)   
% 
% ---------------------------------------------------------------------------
% INPUT
% ---------------------------------------------------------------------------
% - objlss is a large-scale system
% - N is the prediction horizon
% - flag is an optional parameter
%    .ExoBounded specify indices of exogenous inputs that must be
%       considered as disturbances (default is empty)
% - options from YALMIP, i.e. sdpsettings object (optional)
%
% ---------------------------------------------------------------------------
% OUTPUT
% ---------------------------------------------------------------------------
% ctrl is a centralized MPC regulator for the given large-scale system
%
% ---------------------------------------------------------------------------
% PROPERTIES
% ---------------------------------------------------------------------------
% Most important properties.
% A, B, M : dynamic of the lss
% N : prediction horizon
% Hx, Kx, Hu, Ku : constraints for x and u
% Z : parameterizedRCI
% 
%
% ---------------------------------------------------------------------------
% METHODS
% ---------------------------------------------------------------------------
% URH: compute the control action
% XFQPMAX: design an ellipsoidal terminal set and a quadratic cost function
% XFQPMAXDEC: design a block-diagonal ellipsoidal terminal set and quadratic 
%             cost function
% ZEROTERMINAL: design a quadratic cost function, without terminal penalty
%               and terminal set as the set-point 
%
    
% Copyright is with the following author(s):
%
% (C) 2014 
%     Stefano Riverso, Giancarlo Ferrari Trecate
%     Identification and Control of Dynamic Systems Laboratory,
%     Universita' degli Studi di Pavia, Italy
%     stefano.riverso(at)unipv.it, giancarlo.ferrari(at)unipv.it
% (C) 2012 
%     Stefano Riverso, Alberto Battocchio, Giancarlo Ferrari Trecate
%     Identification and Control of Dynamic Systems Laboratory,
%     Universita' degli Studi di Pavia, Italy
%     stefano.riverso(at)unipv.it, alberto.battocchio01(at)universitadipavia.it,
%     giancarlo.ferrari(at)unipv.it
%

% ---------------------------------------------------------------------------
% Legal note:
%          This program is free software; you can redistribute it and/or
%          modify it under the terms of the GNU General Public
%          License as published by the Free Software Foundation; either
%          version 2.1 of the License, or (at your option) any later version.
%
%          This program is distributed in the hope that it will be useful,
%          but WITHOUT ANY WARRANTY; without even the implied warranty of
%          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%          General Public License for more details.
% 
%          You should have received a copy of the GNU General Public
%          License along with this library; if not, write to the 
%          Free Software Foundation, Inc., 
%          59 Temple Place, Suite 330, 
%          Boston, MA  02111-1307  USA
%
% ---------------------------------------------------------------------------
    
    properties ( SetAccess = private , Hidden = false )
        N              ;    % prediction horizon
        A              ;    % local dynamics
        B              ;    % local dynamics
        Hx             ;    % state constraints
        Kx             ;    % state constraints
        Hu             ;    % input constraints
        Ku             ;    % input constraints
        M              ;    % local dynamics
        Hd             ;    % exogenous inputs constraints
        Kd             ;    % exogenous inputs constraints
        Z              ;    % parameterizedRCI if ExoBounded is not empty
        Exo            ;    % indeces of exogenous inputs
        ExoBounded     ;    % indeces of exogenous inputs as disturbances
    end
    
    properties ( SetAccess = private , Hidden = true )
        stateCons      ;    % anonymous function for state constraints
        inputCons      ;    % anonymous function for input constraints
        initialCons    ;    % anonymous function for initial constraints for pnp
        dynState       ;    % anonymous function for dynamics
        TerminalSet    ;    % flag for defined terminal set
        CostFunction   ;    % flag for defined cost function
        J              ;    % anonymous function for cost function
        Xf             ;    % anonymous function for terminal set
        ni             ;    % states for each subsystem
        mi             ;    % control inputs for each subsystem   
    end
    
    
    
    methods
        function ctrl = cenmpc(objlss,N,flag,options)
            
            global pnpmpcToolboxGlobalOptions;
            if ~isstruct(pnpmpcToolboxGlobalOptions),
                pnpmpc_toolbox_err;
            end
            
            if nargin==0
                return;
            end
            
            yalmip('clear')
            
            if nargin>4 || nargin<2
                error('CENMPC: parameters in input. see help CENMPC');
            end
            
            
            if ~isa(objlss,'lss')
                error('CENMPC: first argument must be a lss object')
            end
            
            if isempty(objlss.Ts)
                error('CENMPC: lss object must be a discrete-time system')
            end
            
            if ~isempty(objlss.C)
                warning('CENMPC class will design a state-feedback centralized MPC controller')
            end
            
            if ~(N>0)
                error('CENMPC: N (receding horizon) must be positive')
            end
            
            if ( isempty(objlss.B) || all(all(objlss.B==0)) ) && ( isempty(objlss.Bcen) || all(all(objlss.Bcen==0)) )
                error('CENMPC: lss is an autonomous system')
            end
            
            if nargin~=4 || isempty(options)
                options = sdpsettings;
            end
            
            ctrl.N       = N;
            ctrl.A       = full(objlss.A);
            ctrl.B       = [ full(objlss.B) full(objlss.Bcen) ];
            ctrl.M       = full(objlss.M);
            ctrl.Hd      = full(objlss.Hd);
            ctrl.Kd      = full(objlss.Kd);
            ctrl.Hx      = objlss.Hx;
            ctrl.Kx      = objlss.Kx;
            ctrl.Hu      = [ objlss.Hu zeros(size(objlss.Hu,1),objlss.numICen) ; zeros(size(objlss.Hcen,1),size(objlss.Hu,2)) objlss.Hcen ];
            ctrl.Ku      = [ objlss.Ku ; objlss.Kcen ];
            n            = size(ctrl.A,2);
            m            = size(ctrl.B,2);
            ctrl.ni      = objlss.ni;
            ctrl.mi      = objlss.mi;            
            
            
            HX           = full(ctrl.Hx);
            KX           = full(ctrl.Kx);
            if isempty(HX)
                HX = [eye(n);-eye(n)];
                KX = 1e6*ones(2*n,1);
            end
            HU           = full(ctrl.Hu);
            KU           = full(ctrl.Ku);
            
            HX           = HX./repmat(KX,1,size(HX,2));
            KX           = ones(size(HX,1),1);
            HU           = HU./repmat(KU,1,size(HU,2));
            KU           = ones(size(HU,1),1);
            
            %% check flag
            if nargin<=2 || isempty(flag)
                flag.ExoBounded = [];
            end
            
            %% check exogenous inputs that are disturbances
            if ~isfield(flag,'ExoBounded')
                ctrl.ExoBounded = [];
                ExoBoundedIndex = [];
            else
                [ ctrl.ExoBounded , ExoBoundedIndex ] = intersect(1:objlss.numExo,unique(flag.ExoBounded));
                ctrl.ExoBounded = reshape(ctrl.ExoBounded,1,[]);
            end
            
            [ ctrl.Exo , diffExo ] = setdiff(1:objlss.numExo,ctrl.ExoBounded);
            ctrl.Exo = reshape(ctrl.Exo,1,[]);
            
            
            %% if tube-MPC
            if ~isempty(ctrl.ExoBounded)
                %% check if the disturbances are unbounded
                if isempty(ctrl.Hd)
                    error('CENMPC: exogenous inputs are not bounded. It is impossible design robust centralized MPC controller')
                else
                    DDset = Polyhedron(full(ctrl.Hd),full(ctrl.Kd)).projection(ExoBoundedIndex);
                    HDD   = DDset.A;
                    KDD   = DDset.b;
                end
                [ j ] = ctrl.checkUnbounded( ctrl.M(:,ExoBoundedIndex) , HDD , KDD );
                if ~isempty(j)
                    error(['CENMPC: exogenous input ' num2str(ctrl.ExoBounded(ExoBoundedIndex(j))) ' is unbounded']);
                end
                
                kmin = ctrbindex(ctrl.A,ctrl.B);
                kmax = 10*kmin;
                
                [ vertexW ] = ctrl.computeExtreme( objlss.M(:,ExoBoundedIndex) , HDD , KDD );
                
                % try different k
                for k=kmin:kmax
                    testZi = 1;
                    % compute RCI Zi
                    try
                        Zi = parameterizedRCI(ctrl.A,ctrl.B,k,{HX,KX},{HU,KU},{vertexW,0},zeros(n,1),zeros(m,1),options);
                    catch e
                        warning(e.message)
                        continue;
                    end
                    
                    % in the following, the code check numerical problems
                    if Zi.info.problem==0
                        testZi = 0;
                        if Zi.isinside(zeros(size(ctrl.A,1),1),options)~=1
                            testZi = 1;
                            continue;
                        end
                        ctrl.Z    = Zi;
                        
                        bxrep    = repmat(ctrl.Z.bx,1,size(ctrl.Z.x0,1));
                        xV       = (1-ctrl.Z.alpha)^-1*( bxrep + ctrl.Z.x0' );
                        Hxctmp.A    = repmat(HX,size(xV,2),1);
                        if ~isempty(Hxctmp.A)
                            Hxctmp.B = repmat(KX,size(xV,2),1)-sum(Hxctmp.A.*reshape(repmat(xV,size(HX,1),1),size(xV,1),[])',2);
                            Hxctmp   = cddmex('reduce_h',Hxctmp);
                            ctrl.Hx  = Hxctmp.A;
                            ctrl.Kx  = Hxctmp.B;
                            % compute tightened constraints X for tube controller
                            for i = 1:ctrl.Z.k-1
                                xV       =  (1-ctrl.Z.alpha)^-1*( bxrep + ctrl.Z.x(:,:,i) );
                                Hxctmp.A =  repmat(ctrl.Hx,size(xV,2),1);
                                Hxctmp.B =  repmat(ctrl.Kx,size(xV,2),1)-sum(Hxctmp.A.*reshape(repmat(xV,size(ctrl.Hx,1),1),size(xV,1),[])',2);
                                Hxctmp   =  cddmex('reduce_h',Hxctmp);
                                ctrl.Hx  =  Hxctmp.A;
                                ctrl.Kx  =  Hxctmp.B;
                            end
                            if any(ctrl.Hx*zeros(size(ctrl.A,1),1)>ctrl.Kx)
                                testZi = 1;
                                continue;
                            end
                        end
                        
                        % compute tightened constraints U for tube controller
                        ctrl.Hu   = HU;
                        ctrl.Ku   = KU;
                        if ~isempty(ctrl.Hu)
                            burep    = repmat(ctrl.Z.bu,1,size(ctrl.Z.x0,1));
                            for i = 1:ctrl.Z.k
                                uV       =  (1-ctrl.Z.alpha)^-1*( burep + ctrl.Z.u(:,:,i) );
                                Hvtmp.A  =  repmat(ctrl.Hu,size(uV,2),1);
                                Hvtmp.B  =  repmat(ctrl.Ku,size(uV,2),1)-sum(Hvtmp.A.*reshape(repmat(uV,size(ctrl.Hu,1),1),size(uV,1),[])',2);
                                Hvtmp    =  cddmex('reduce_h',Hvtmp);
                                ctrl.Hu   =  Hvtmp.A;
                                ctrl.Ku   =  Hvtmp.B;
                            end
                            if any(ctrl.Hu*zeros(size(ctrl.B,2),1)>ctrl.Ku)
                                testZi = 1;
                                continue;
                            end
                        end
                        break;
                    end
                end
                
                if testZi==1
                    error('CENMPC: it is impossible to design a robust MPC controller.');
                end
                
                % x0-xc(0) is inside the RCI
                ctrl.initialCons = @(x0,xc,lambda) ( [ lambda(:)>=0 ; sum(lambda,2) == 1 ; x0-xc == (1-ctrl.Z.alpha)^-1*( ctrl.Z.bx + sum(repmat(reshape(lambda',1,[]),size(ctrl.A,1),1).*[ctrl.Z.x0' reshape(ctrl.Z.x(:,:,1:ctrl.Z.k-1),size(ctrl.Z.A,1),[])],2)) ] );
                
            else
                
                
                ctrl.initialCons = @(x0,xc,lambda)( x0-xc(:,1)==0 );
                
                
            end
            
            
            
            ctrl.A   = sparse(ctrl.A);
            ctrl.B   = sparse(ctrl.B);
            ctrl.M   = sparse(ctrl.M);
            
            %% in dynState (dynamic of the state) we receive xc(:)
            AA = [];
            BB = [];
            for i=1:N
                AA = blkdiag(AA,ctrl.A);
                BB = blkdiag(BB,ctrl.B);
            end
            
            if isempty(ctrl.Exo)
                ctrl.dynState = @(xc,v,d)(xc(n+1:end) == AA*xc(1:end-n)+BB*v);
            else
                MM = [];
                for i=1:N
                    MM = blkdiag(MM,objlss.M(:,diffExo));
                end
                ctrl.dynState = @(xc,v,d)( xc(n+1:end) == AA*xc(1:end-n)+BB*v+MM*d );
            end
            
            if isempty(objlss.Hx)
                ctrl.Hx = [];
                ctrl.Kx = [];
            else
                ctrl.Hx = sparse(ctrl.Hx);
                ctrl.Kx = sparse(ctrl.Kx);
            end
            if isempty(objlss.Hu)
                ctrl.Hu = [];
                ctrl.Ku = [];
            else
                ctrl.Hu = sparse(ctrl.Hu);
                ctrl.Ku = sparse(ctrl.Ku);
            end
            ctrl.Hd  = sparse(ctrl.Hd);
            ctrl.Kd  = sparse(ctrl.Kd);
            
            
            %% constraints on x
            if ~isempty(ctrl.Hx)
                Hbx = repmat(ctrl.Hx,N,1);
                Kbx = repmat(ctrl.Kx,N,1);
                ctrl.stateCons = @(xc)(sum(Hbx.*(reshape(repmat(xc(:,1:N),size(ctrl.Hx,1),1),n,[]))',2)<=Kbx);
            else
                ctrl.stateCons = @(xc)([]);
            end
            
            %% constraints on u
            if ~isempty(ctrl.Hu)
                Hbu = sparse(repmat(ctrl.Hu,N,1));
                Kbu = sparse(repmat(ctrl.Ku,N,1));
                ctrl.inputCons = @(v)(sum(Hbu.*(reshape(repmat(v,size(ctrl.Hu,1),1),m,[]))',2)<=Kbu);
            else
                ctrl.inputCons = @(v)([]);
            end
            
            %% Terminal Set and Cost Function are created with own methods
            ctrl.TerminalSet  = false;
            ctrl.CostFunction = false;
            
            yalmip('clear')
            
        end
        
    end
    
    
    
    
    
    
    methods(Static)
        
        
        function [ vertexW ] = computeExtreme( matrixD , HmatrixD , KmatrixD )
            %% compute extreme of flat polytope
            matrixD = full(matrixD);
            
            tol     = 1e-6;
            
            P   = Polyhedron(full(HmatrixD),full(KmatrixD));
            tmp = P.V*matrixD';
            tmp(abs(tmp)<=tol) = 0;
            tmp = sortrows(tmp);
            tmp = tmp([true; any(abs(diff(tmp))>tol,2)],:);
            P   = Polyhedron(tmp);
            if size(tmp,1)<=500
                P = P.computeHRep;
                if ~isempty(P.Ae)
                    P   = Polyhedron([P.A;P.Ae;-P.Ae],[P.b;P.be+1e-6;-P.be+1e-6]).minVRep;
                end
                tmp = cddmex('extreme',struct('A',[eye(P.Dim);-eye(P.Dim)],'B',tol*ones(2*P.Dim,1)));
                P   = Polyhedron([P.V;tmp.V]);
                vertexW = P.minVRep.V;
            else
                warning('CENMPC: numerical simplifications for the set of bounded disturbances.')
                P = P.outerApprox;
                vertexW = P.V;
            end         
        end
        
        
        
        
        function [ j ] = checkUnbounded( matrix , Hmatrix , Kmatrix )
            %% check if the variables are unbounded
            j = [];
            big = 10^7;
            Hmatrix = [eye(size(matrix,2));-eye(size(matrix,2));Hmatrix];
            Kmatrix = [ones(2*size(matrix,2),1)*big;Kmatrix];
            if ~isempty(Hmatrix)
                for i = 1:size(Hmatrix,2)
                    objective    = zeros(1,size(Hmatrix,2));
                    objective(i) = 1;
                    INmin        = struct('obj',objective,'A',Hmatrix,'B',Kmatrix);
                    OUTmin       = cddmex('solve_lp',INmin);
                    objective(i) = -1;
                    INmax        = struct('obj',objective,'A',Hmatrix,'B',Kmatrix);
                    OUTmax       = cddmex('solve_lp',INmax);
                    if OUTmin.xopt(i)==-big || OUTmax.xopt(i)==big
                        if any(matrix(:,i)~=0)
                            j = i;
                            break
                        end
                    end
                end
            end
        end
        
        
    end
    
    
    
    
end