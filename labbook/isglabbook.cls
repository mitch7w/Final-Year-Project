%%
%%  ISG@UP Lab Book Document Class.
%%  Copyright (C) 2014-2015, University of Pretoria.
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{isglabbook}[2015/03/02 v1.5 ISG@UP Lab Book Document Class]

%% This class is an extension of the standard report class and will pass
%%   all unknown options to the foundation class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper]{report}

%% Establish input relative to a directory.
\RequirePackage{import}

%% Use the "geometry" package to maximize page usage whilst retaining enough
%%   margin space for staples and editorial notes.
\RequirePackage[a4paper,includeheadfoot,
                top=20mm,inner=20mm,outer=20mm,bottom=20mm
               ]{geometry}

%% Load support for colour.
\RequirePackage[table]{xcolor}

% COLORS (Tango) Mostly by Philip Bunge
% http://pbunge.crimson.ch/
\definecolor{LightGray}{gray}{0.95}
\definecolor{White}{gray}{1.0}
\definecolor{Black}{gray}{0.0}
\definecolor{LightButter}{rgb}{0.98,0.91,0.31}
\definecolor{LightOrange}{rgb}{0.98,0.68,0.24}
\definecolor{LightChocolate}{rgb}{0.91,0.72,0.43}
\definecolor{LightChameleon}{rgb}{0.54,0.88,0.20}
\definecolor{LightSkyBlue}{rgb}{0.45,0.62,0.81}
\definecolor{LightPlum}{rgb}{0.68,0.50,0.66}
\definecolor{LightScarletRed}{rgb}{0.93,0.16,0.16}
\definecolor{Butter}{rgb}{0.93,0.86,0.25}
\definecolor{Orange}{rgb}{0.96,0.47,0.00}
\definecolor{Chocolate}{rgb}{0.75,0.49,0.07}
\definecolor{Chameleon}{rgb}{0.45,0.82,0.09}
\definecolor{SkyBlue}{rgb}{0.20,0.39,0.64}
\definecolor{Plum}{rgb}{0.46,0.31,0.48}
\definecolor{ScarletRed}{rgb}{0.80,0.00,0.00}
\definecolor{DarkButter}{rgb}{0.77,0.62,0.00}
\definecolor{DarkOrange}{rgb}{0.80,0.36,0.00}
\definecolor{DarkChocolate}{rgb}{0.56,0.35,0.01}
\definecolor{DarkChameleon}{rgb}{0.30,0.60,0.02}
\definecolor{DarkSkyBlue}{rgb}{0.12,0.29,0.53}
\definecolor{DarkPlum}{rgb}{0.36,0.21,0.40}
\definecolor{DarkScarletRed}{rgb}{0.64,0.00,0.00}
\definecolor{Aluminium1}{rgb}{0.93,0.93,0.92}
\definecolor{Aluminium2}{rgb}{0.82,0.84,0.81}
\definecolor{Aluminium3}{rgb}{0.73,0.74,0.71}
\definecolor{Aluminium4}{rgb}{0.53,0.54,0.52}
\definecolor{Aluminium5}{rgb}{0.33,0.34,0.32}
\definecolor{Aluminium6}{rgb}{0.18,0.20,0.21}

%% Enable more advanced character protrusion and font expansion, adjustment
%%   of inter-word spacing and additional kerning, etc.
\RequirePackage[activate={true,nocompatibility},tracking=true,
    kerning=true,spacing=true]{microtype}
\microtypecontext{spacing=nonfrench}

%% Load modern Latin font encoding template T1, as well as additional
%%   symbols.
\RequirePackage{lmodern}
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\RequirePackage{fix-cm}

%% Use the standard PostScript Type 1 version fonts for Roman, Sans Serif
%%   and Typewriter font classes. This allows the PDF files to be smaller
%%   as the custom TeX fonts do not have to be embedded.
\RequirePackage{mathptmx}
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage{courier}

%% Enable fancy PostScript based characters.
\RequirePackage{pifont}

%% Load AMS mathematics support, fonts and symbols.
\RequirePackage[fleqn]{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amstext}
\RequirePackage{amssymb}
\RequirePackage{mathtools}

%% Set maths equation number to reset per chapter
\numberwithin{equation}{chapter}

%% Yet more mathematics font related modules...
\RequirePackage{bm}
\RequirePackage[mathcal]{eucal}
\RequirePackage{upgreek}

%% Load additional symbols.
\RequirePackage{wasysym}
\RequirePackage{latexsym}
%\RequirePackage{MnSymbol}

%% Define input language as British English.
\RequirePackage[british]{babel}

%% Add calculator for dimensions.
\RequirePackage{calc}

%% Load support for advanced control structures in macros.
\RequirePackage{ifthen}

%% Load support for direct PDF generation.
\RequirePackage{ifpdf}

%% Special spaces in macros.
\RequirePackage{xspace}

%% Enable multiple column support
\RequirePackage{multicol}

%% Load support for external PDF file inclusion.
\RequirePackage[final]{pdfpages}


%% Modify section titles and the spacing above/below them, resulting
%%   in significant space-savings. The nobottomtitles command ensures
%%   that no titles appear at the end of a page.
\RequirePackage[small,compact,nobottomtitles*]{titlesec}
\renewcommand{\bottomtitlespace}{.12\textheight}
\titleformat{\chapter}[frame]{\cmbrfamily\Huge\bfseries}{\thechapter}{20pt}{\Huge}
\titleformat*{\section}{\cmbrfamily\Large\bfseries}
\titleformat*{\subsection}{\cmbrfamily\large\bfseries}
\titleformat*{\subsubsection}{\cmbrfamily\bfseries}
%\RequirePackage{titletoc}
%\titlecontents{section}[0pt]{\filright}
%              {\contentspush{\thecontentslabel\ }}
%              {}{\titlerule*[8pt]{.}\contentspage}

%% To allow inclusive page numbering.
\RequirePackage{lastpage}

%% Enable the use of Hypertext references.
\RequirePackage[
  pdftex,                       % Hyperlinked references for PDF output
  backref=page,                 % Include backref to page from citation
  breaklinks=true,              % Break links to fit in page width
  bookmarks=true,               % Enable bookmarks
  bookmarksnumbered=true,       % Enable add section numbers to bookmarks
  colorlinks=true,              % Coloured links, no boxes
  linkcolor=blue,               % Links are set in blue
  pdffitwindow=false,           % Do not fit window to page
  pdfpagelabels=true,           % Embed proper page labels in PDF
  pdfpagemode=UseOutlines,      % Start with bookmarks open
  pdfnewwindow=true,            % Open links in new window
  pdfstartview=FitH,            % Fits width of page to window
  pdfdisplaydoctitle]{hyperref}
\RequirePackage{breakurl}

%% Fix references to captions so that the reference is to the top of the
%%   figure, ensuring that it is visible when clicked.
%\RequirePackage[all]{hypcap}

%% Enable support for extended list environments.
\RequirePackage{paralist}

%% Enable shadow boxes.
\RequirePackage{shadow}

%% Enable underline support.
\RequirePackage{soul}
\setstcolor{ScarletRed}

%% Load support for extended floating environments.
\RequirePackage{float}

%% Packages to allow us to rotate things.
\RequirePackage{rotating}
\RequirePackage{rotfloat}

%% Enable hanging captions.
\RequirePackage[small,hang]{caption}
\RequirePackage{subcaption}

%% Allow verbatim file importing
\RequirePackage{alltt}
\RequirePackage{moreverb}

%% Support syntax highlighted program listings
\RequirePackage{listings}
\lstset{
  backgroundcolor=\color{LightGray}, %
  basicstyle=\color{Black}\ttfamily{}, %
  keywordstyle=[1]\color{DarkSkyBlue}, %
  keywordstyle=[2]\color{DarkScarletRed}, %
  keywordstyle=[3]\bfseries{}, %
  keywordstyle=[4]\color{DarkPlum}, %
  keywordstyle=[5]\color{SkyBlue}, %
  commentstyle=\color{Aluminium4}, %
  stringstyle=\color{Chocolate}, %
  identifierstyle=\color{Black}, %
  emphstyle=\color{Black}, %
  numbers=left, %
  stepnumber=1, %
  frame=tb, %
  captionpos=t, %
  lineskip=\smallskipamount{}, %
  aboveskip=\bigskipamount{}, %
  belowskip=\medskipamount{}, %
  commentstyle=\itshape\small{}, %
  tabsize=2, %
  breaklines=false, %
  rulecolor=\color{Black!30}, %
  showspaces=false, %
  showstringspaces=false, %
  showtabs=false, %
} % chktex 6

%% Add support for pseudo code for algorithms
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

%% Enable fancy long tables.
\RequirePackage{booktabs}

%% Enable tabular horizontal lines.
\RequirePackage{hhline}

%% Enable index generation.
\RequirePackage{makeidx}

%% Appendix support
\RequirePackage{appendix}

%% Ensure that only one expanded version of an acronym appears.
\RequirePackage[printonlyused,withpage]{acronym}

%% Parametrized enumerations
\RequirePackage{enumerate}

%% Package to support advanced theorem
\RequirePackage{theorem}

%% Enable multiple column / row support
\RequirePackage{multicol}
\RequirePackage{multirow}

%% Include support for formatting of large numbers.
\RequirePackage{seqsplit}

%% Support for date / time operations
\RequirePackage{datetime}
% Switch to ISO date displayed form 2000/03/08.
\yyyymmdddate{}

%% Load package to support standard graphics.
\RequirePackage{graphicx}

%% Load package to support advanced vector graphics.
\RequirePackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,matrix,mindmap,
                shadows,fit,calc,intersections}
\pgfdeclarelayer{background}
\pgfdeclarelayer{foreground}
\pgfsetlayers{background,main,foreground}

%% Required for inserting graphs.
\RequirePackage{pgfplots}
\pgfplotsset{compat=newest}

%% Disable paragraph indenting.
\RequirePackage[parfill]{parskip}

%% Enable multiple figures per line
%\RequirePackage{subfigure}
%\RequirePackage{wrapfig}

%% Required for inserting SVG images.
%\RequirePackage{svg}

%% Lorem ipsum dummy text
\RequirePackage{lipsum}

%% Used for redacted blocks
\RequirePackage{mdframed}
\mdfdefinestyle{redact}{linecolor=ScarletRed,linewidth=0.8pt,
    frametitle={\textcolor{white}{Redacted Content}},
    frametitlerule=true,
    frametitlebackgroundcolor=ScarletRed!70!white}

%% To help make todo lists visible, use the todonotes package.
\RequirePackage[color=blue,backgroundcolor=SkyBlue]{todonotes}
\newcommand{\todoredefined}[2][]
  {\todo[inline, #1]{\color{white}\upshape #2}}

%% Space below captions -- add some leeway to avoid orphan / stray lines
\setlength{\belowcaptionskip}{10pt plus 3pt minus 3pt}
\setlength{\abovecaptionskip}{\the\abovecaptionskip plus 3pt minus 3pt}
\addtolength{\floatsep}{0pt plus 3pt minus 3pt}
\addtolength{\textfloatsep}{0pt plus 3pt minus 3pt}
\addtolength{\intextsep}{0pt plus 3pt minus 3pt}
\widowpenalty=9000
\clubpenalty=9000

% Not going to use margins
\setlength{\marginparwidth}{0 pt}
\setlength{\marginparsep}{0 pt}
\setlength{\evensidemargin}{0 pt}
\setlength{\oddsidemargin}{0 pt}

%% Reset maths style parameters.
\setlength{\abovedisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{0pt plus 1pt}
\setlength{\belowdisplayshortskip}{0pt minus 1pt}

%% Define shorthand font family selection macros
%%                   Computer Concrete font family: \ccrfamily
\newcommand{\ccrfamily} {\fontencoding{T1}\fontfamily{ccr}\selectfont}
%%                   Computer Modern Roman font family: \cmrfamily
\newcommand{\cmrfamily} {\fontencoding{T1}\fontfamily{cmr}\selectfont}
%%                   Computer Modern Sans font family: \cmssfamily
\newcommand{\cmssfamily} {\fontencoding{T1}\fontfamily{cmss}\selectfont}
%%                   Computer Fibonacci font family: \cmfibfamily
\newcommand{\cmfibfamily} {\fontencoding{T1}\fontfamily{cmfib}\selectfont}
%%                   Computer Dunhill font family: \cmdhfamily
\newcommand{\cmdhfamily} {\fontencoding{T1}\fontfamily{cmdh}\selectfont}
%%                   Computer Modern Bright font family: \cmbrfamily
\newcommand{\cmbrfamily} {\fontencoding{T1}\fontfamily{cmbr}\selectfont}

%%                   Latin Modern font family: \lmrfamily
\newcommand{\lmrfamily} {\fontencoding{T1}\fontfamily{lmr}\selectfont}
%%                   Latin Modern Sans font family: \lmssfamily
\newcommand{\lmssfamily} {\fontencoding{T1}\fontfamily{lmss}\selectfont}

%%                   Pandora Roman font family: \panrfamily
\newcommand{\panrfamily} {\fontencoding{T1}\fontfamily{panr}\selectfont}
%%                   Pandora Sans font family: \pssfamily
\newcommand{\pssfamily} {\fontencoding{T1}\fontfamily{pss}\selectfont}

%%                   Postscript font: Times family: \ptmfamily
\newcommand{\ptmfamily} {\fontencoding{T1}\fontfamily{ptm}\selectfont}
%%                   Postscript font: Palatino family: \pplfamily
\newcommand{\pplfamily} {\fontencoding{T1}\fontfamily{ppl}\selectfont}
%%                   Postscript font: New Century family: \pncfamily
\newcommand{\pncfamily} {\fontencoding{T1}\fontfamily{pnc}\selectfont}
%%                   Postscript font: Bookman family: \pbkfamily
\newcommand{\pbkfamily} {\fontencoding{T1}\fontfamily{pbk}\selectfont}
%%                   Postscript font: Helvetica family: \phvfamily
\newcommand{\phvfamily} {\fontencoding{T1}\fontfamily{phv}\selectfont}
%%                   Postscript font: Avant Garde family: \pagfamily
\newcommand{\pagfamily} {\fontencoding{T1}\fontfamily{pag}\selectfont}
%%                   Postscript font: Courier family: \pcrfamily
\newcommand{\pcrfamily} {\fontencoding{T1}\fontfamily{pcr}\selectfont}
%%                   Postscript font: Utopia family: \putfamily
\newcommand{\putfamily} {\fontencoding{T1}\fontfamily{put}\selectfont}
%%                   Postscript font: Charter family: \bchfamily
\newcommand{\bchfamily} {\fontencoding{T1}\fontfamily{bch}\selectfont}

%% Define new font shapes:
%%                   Outline: \olshape
\newcommand{\olshape} {\fontshape{ol}\selectfont}
%%                   Upright italic: \uishape
\newcommand{\uishape} {\fontshape{ui}\selectfont}

%% Define slanted formatting for Latin phrases.
%%                   a posteriori:    \apost
\newcommand{\apost} {\emph{a posteriori}\xspace}
%%                   a prior:    \aprior
\newcommand{\apriori} {\emph{a priori}\xspace}
%%                   c.:    \circa
\newcommand{\circa} {\emph{c.}\xspace}
%%                   cf.:    \compare
\newcommand{\compare} {\emph{cf.}\xspace}
%%                   crica:    \circal
\newcommand{\circal} {\emph{circa}\xspace}
%%                   e.g.:      \eg
\newcommand{\eg} {\emph{e.g.},\xspace}
%%                   etc.:      \etc
\newcommand{\etc} {\emph{etc.}\@\xspace} % \etc
%%                   et al.:    \etal
\newcommand{\etal} {\emph{et al.}\xspace} % \etal
%%                   et seq.:    \etseq
\newcommand{\etseq} {\emph{et seq.}\xspace}
%%                   i.e.:      \ie
\newcommand{\ie} {\emph{i.e.}\xspace}

%% Miscellaneous
\DeclareRobustCommand{\BibTeX}{\textsc{Bib}\TeX\ }

%% Mathematical Units/Symbols
\newcommand{\leftnorm}{\left\|}
\newcommand{\rightnorm}{\right\|}
\newcommand{\Celsius}{{\ensuremath{\;^{\circ}C}}}    %% Degrees Celsius
\newcommand{\degree}{^{\circ}}
\newcommand{\tr}{{\ensuremath{\operatorname{tr}}}}       %% matrix trace operator
\newcommand{\var}{{\ensuremath{\operatorname{var}}}}     %% variance operator
\newcommand{\sd}{{\ensuremath{\operatorname{sd}}}}       %% standard deviation operator
\newcommand{\mode}{\ensuremath{\operatorname{mode}}}
\newcommand{\m}[1]{\ensuremath{\mathbf{#1}}}
\newcommand{\mt}[1]{\ensuremath{\mathbf{#1}^T}}
\newcommand{\median}[1]{\ensuremath{\underset{#1}{\operatorname{median}}}}
\newcommand{\argmin}[1]{\ensuremath{\underset{#1}{\operatorname{argmin}}}}
\newcommand{\argmax}[1]{\ensuremath{\underset{#1}{\operatorname{argmax}}}}
\newcommand{\expect}[1]{\ensuremath{E[#1]}}
\newcommand{\uv}[1]{\ensuremath{\underline{#1}}}
\newcommand{\iv}{\uv{i}}
\newcommand{\jv}{\uv{j}}
\newcommand{\kv}{\uv{k}}

%% TradeMark/Copyright etc. symbols
\newcommand{\Copyright}{\Pisymbol{psy}{227}\xspace} %% Registered Name
\newcommand{\RegMark}{\Pisymbol{psy}{226}\xspace} %% Registered Name
\newcommand{\TradeMark}{\hspace{1pt}\Pisymbol{psy}{228}\xspace}   %% Registered TradeMark

%% Define nice end of document marker
\newcommand{\nendsign} {
  \vskip 24pt
  \centerline{$\Longleftarrow\bowtie\Longrightarrow$}
} % \nendsign

%% Define nice end of chapter sign.
\newcommand{\pendsign} {
  \vspace{24pt}
  \centerline{$\longleftarrow\asymp\longrightarrow$}
  \newpage
} % \pendsign

%% Define nice end of concept sign.
\newcommand{\cendsign} {
  \vspace{1em}
  \centerline{$\longleftarrow\asymp\longrightarrow$}
  \vspace{1em}
} % \cendsign

%% Change format of enumerated lists
%\renewcommand{\theenumi}{\roman{enumi}}
%\renewcommand{\labelenumi}{\theenumi).}

%% Make nice looking Table and Figure captions.
\renewcommand{\captionlabelfont}{\bfseries\sffamily}

%% Define a new Table of Contents like file for Algorithms
%\newcommand{\listofalgorithms}{\newpage\chapter*{List of Algorithms}\@starttoc{loa}}
\newcommand{\acaption}[1]{\addcontentsline{loa}{algorithm}{#1}}
\newcommand{\l@algorithm}{\@dottedtocline{1}{1.5em}{3.2em}}

%% Define the environment for Algorithms
\floatstyle{ruled}
\newfloat{algorithm}{tb}{loa}[section]
\floatname{algorithm}{Algorithm}
\newcommand{\AlgRef}[1]{Alg.~\textbf{$\lfloor$\ref{#1}$\rfloor$}}
\newcommand{\AlgNRef}[1]{\textbf{$\lfloor$\ref{#1}$\rfloor$}}

%% Define a new Table of Contents like file for Examples
%\newcommand{\listofexamples}{\newpage\chapter*{List of Examples}\@starttoc{loe}}
\newcommand{\ecaption}[1]{\addcontentsline{loe}{example}{#1}}
\newcommand{\l@example}{\@dottedtocline{1}{1.5em}{3.2em}}

%% Define the environment for Algorithms
\floatstyle{boxed}
\newfloat{example}{!hbt}{loe}[section]
\floatname{example}{Example}
\newcommand{\ExpRef}[1]{Example~\textbf{$\lceil$\ref{#1}$\rceil$}}
\newcommand{\ExpNRef}[1]{\textbf{$\lceil$\ref{#1}$\rceil$}}

%% Increase the number widths in the tables
\renewcommand{\l@section}{\@dottedtocline{1}{1.5em}{3.2em}}
\renewcommand{\l@subsection}{\@dottedtocline{2}{4.7em}{3.4em}}
\renewcommand{\l@subsubsection}{\@dottedtocline{3}{8.1em}{3.6em}}
\renewcommand{\l@figure}{\@dottedtocline{1}{1.5em}{3.2em}}
\renewcommand{\l@table}{\l@figure}
%\setcounter{secnumdepth}{-2}
\setcounter{secnumdepth}{0}

%% Change floats to be more realistic
\clubpenalty=9999
\widowpenalty=9999
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.2}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}

%% Verbatim size
\newlength{\verbatimizeup}
\setlength{\verbatimizeup}{-\parskip}%
\addtolength{\verbatimizeup}{-\parskip}%
\addtolength{\verbatimizeup}{-10pt}

%% Lengths for itemized environments
\newlength{\itemizeup}
\setlength{\itemizeup}{-\parskip}%
\addtolength{\itemizeup}{-5pt}
\setlength{\itemsep}{2pt}

%% Set new shadow box parameters
%\renewcommand{\fboxsep}{20pt}
%\renewcommand{\sboxsep}{20pt}
%\renewcommand{\shadowsize}{2pt}
%\renewcommand{\sdim}{2\fboxsep}

%% Create new indexing command
\newcommand{\aidx}[1]{\ac{#1}\index{#1}}
\newcommand{\idx}[1]{{#1}\index{#1}}
\newcommand{\idxl}[1]{{#1}\index{\lowercase{#1}}}
\newcommand{\bidx}[1]{{#1}\index{#1|textbf}}
\newcommand{\bidxl}[1]{{#1}\index{\lowercase{#1}|textbf}}
\newcommand{\iidx}[1]{{#1}\index{#1|textit}}
\newcommand{\iidxl}[1]{{#1}\index{\lowercase{#1}!textit}}
\newcommand{\sidx}[1]{{#1}\index{#1|textsl}}
\newcommand{\sidxl}[1]{{#1}\index{\lowercase{#1}|textsl}}
\newcommand{\cidx}[1]{{#1}\index{#1|textsc}}
\newcommand{\cidxl}[1]{{#1}\index{\lowercase{#1}|textsc}}
\newcommand{\tidx}[1]{{#1}\index{#1|texttt}}
\newcommand{\tidxl}[1]{{#1}\index{\lowercase{#1}|texttt}}

%% Define dot types to be used in graphs to indicate sample points
\newcommand{\CHo}{\makebox(0,0){$\bullet$}}
\newcommand{\CHd}{\makebox(0,0){$\diamondsuit$}}
\newcommand{\CHc}{\makebox(0,0){$\circ$}}
\newcommand{\CHs}{\makebox(0,0){$\star$}}

%% Define commands used to highlight references.
\newcommand{\PrtRef}[1]{Part~\ref{#1}}
\newcommand{\ApxRef}[1]{Appendix~\ref{#1}}
\newcommand{\SecRef}[1]{Section~\ref{#1}}
\newcommand{\PagRef}[1]{page~\pageref{#1}}

\newcommand{\FigRef}[1]{Fig.~\textbf{[\ref{#1}]}}
\newcommand{\FigNRef}[1]{\textbf{[\ref{#1}]}}
\newcommand{\TabRef}[1]{Table~\textbf{\{\ref{#1}\}}}
\newcommand{\TabNRef}[1]{\textbf{\{\ref{#1}\}}}
\newcommand{\EquRef}[1]{Eq.~\textbf{(\ref{#1})}}
\newcommand{\EquNRef}[1]{\textbf{(\ref{#1})}}
\newcommand{\SchRef}[1]{Schematic~\textbf{$<$\ref{#1}$>$}}
\newcommand{\SchNRef}[1]{\textbf{$<$\ref{#1}$>$}}

%% Special reference format used to reference locally available
%%   sources so that, for example, Acrobat will automatically open
%%   the link is clicked.
\newcommand{\lref}[1]{[LOCAL: \href{#1}{\texttt{#1}}]}
\newcommand{\eref}[1]{UP Library EBOOK \href{#1}{link}.}
\newcommand{\ebref}[1]{[EBOOK: \href{#1}{\texttt{E-access}}]}

%% Program management macros
\newcommand{\ProgCodeBegin}{\scriptsize
                            \renewcommand{\baselinestretch}{0.8} }
\newcommand{\ProgCodeEnd}{\renewcommand{\baselinestretch}{1.0}
                          \normalsize}

%%
\renewcommand{\sectionautorefname}{Section}
\renewcommand{\subsectionautorefname}{Section}
\renewcommand{\subsubsectionautorefname}{Section}

%% Start each section on a new page, typical of physical lab books.
\newcommand{\sectionbreak}{\clearpage}

%-----------------------------------------------------------------------------
% NEW ENVIRONMENTS
%-----------------------------------------------------------------------------

%% Create a new environment for the index to avoid the loss of page headers /
%%   footers, the default for the standard version.
\renewenvironment{theindex}
  {\ifthenelse{\boolean{@twocolumn}}{\setboolean{@restonecol}{false}}%
                                    {\setboolean{@restonecol}{true}}%
   \setlength\columnseprule{0pt}%
   \setlength\columnsep{35pt}%
   \small\cmbrfamily%
   \twocolumn[\section*{\indexname}]%
   \markboth{\MakeUppercase\indexname}{\MakeUppercase\indexname}%
   \setlength\parindent{0pt}%
   \setlength\parskip{0pt plus 0.3pt}%
   \let\item\@idxitem }
  {\ifthenelse{\boolean{@restonecol}}{\onecolumn}{\clearpage}}

%% Environment for notes, typeset bold slanted.
\newenvironment{boldslantnote}%
  {\begin{quote}%
     \renewcommand{\baselinestretch}{1}
     \bfseries\slshape\phvfamily}%
  {\end{quote}}

%% Environment for notes, typeset bold.
\newenvironment{boldnote}%
  {\begin{quote}%
     \renewcommand{\baselinestretch}{1}
     \bfseries\small\phvfamily}%
  {\end{quote}}

%% Environment for notes, typeset slanted.
\newenvironment{slantnote}%
  {\begin{quote}%
     \renewcommand{\baselinestretch}{1}
     \slshape\small\phvfamily}%
  {\end{quote}}

%% Environment for notes.
\newenvironment{normalnote}%
  {\begin{quote}%
     \renewcommand{\baselinestretch}{1}
     \small\phvfamily}%
  {\end{quote}}

%% New entry type itemized list
\newenvironment{Ventry}[1]%
  {\begin{list}{}{\renewcommand{\makelabel}[1]{\textsf{##1:}\hfil}%
    \settowidth{\labelwidth}{\textsf{#1:}}%
    \setlength{\leftmargin}{\labelwidth+\labelsep}}}%
  {\end{list}}

%% New entry type itemized list
\newlength{\LentryLen}
\newcommand{\Lentrylabel}[1]{%
  \settowidth{\LentryLen}{\bfseries\cmfamily #1:}%
  \ifthenelse{\lengthtest{\LentryLen > \labelwidth}}%
    {\parbox[b]{\labelwidth}%
      {\makebox[0pt][l]{\bfseries\cmfamily #1:}\\}}%
    {\bfseries\cmfamily #1:}%
  \hfil\relax}
\newenvironment{Lentry}%
  {\renewcommand{\entrylabel}{\Lentrylabel}%
   \begin{entry}}
  {\end{entry}}

%\newenvironment{packed_itemize}{
%  \vspace{\itemizeup}
%  \begin{itemize}
%    \setlength{\itemsep}{2pt}
%    \setlength{\parskip}{0pt}
%    \setlength{\parsep}{0pt}
%}{
%  \end{itemize}
%  \vspace{\itemizeup}
%}

%\newenvironment{packed_enumerate}{
%  \vspace{\itemizeup}
%  \begin{enumerate}
%    \setlength{\itemsep}{2pt}
%    \setlength{\parskip}{0pt}
%    \setlength{\parsep}{0pt}
%}{\end{enumerate}
%  \vspace{\itemizeup}
%}

\newcommand{\newday}[1]{%
    \section*{#1}%
}

\newenvironment{maybe}%
    {\section*{Someday / Maybe}}%
    {\vspace{2mm}\hrule\hspace{\stretch{1}}}

% add "Appendix" to the section heading
\newcommand{\appsection}[1]{\let\oldthesection\thesection%
  \renewcommand{\thesection}{Appendix \oldthesection\ -}
  \section{#1}\let\thesection\oldthesection}
\renewcommand{\appendixpagename}{Appendix}

% Macro for comments / notes on items in lists.
\newcommand{\listnote}[1]{
  {\footnotesize\phvfamily\textbf{Comment:} #1}}

%-----------------------------------------------------------------------------

% Change the way the section mark header looks
%\renewcommand{\sectionmark}[1]{
%  \markright{#1}
%  } % \sectionmark

%% Enable fancy headers, verbatim sections and boxes.
\RequirePackage{fancyhdr}
\RequirePackage{fancyvrb}
\RequirePackage{fancybox}

% Make the header wide enough to fit the header bar
\addtolength{\headheight}{3pt}
\fancyheadoffset[L]{5mm}
\fancyfootoffset[L]{5mm}

%% Redefine the plain style so that \chapter and \part pages look right.
\fancypagestyle{plain}{
  \fancyhf{}                              % Flush all current headers
  \renewcommand{\headrulewidth}{0.4pt}    % Enable the header bar
  \fancyhead[L]{\small\pplfamily\textsc{\doctype}}
  \fancyhead[C]{\lmssfamily\isg}
  \fancyhead[R]{\small\putfamily\textsc{\nouppercase{\rightmark}}}
  \renewcommand{\footrulewidth}{0.4pt}    % Enable the header bar
  \fancyfoot[L]{\scriptsize\phvfamily\docauthor~(\docstudentnumber). \eece. \up}
  \fancyfoot[R]{\footnotesize\putfamily\thepage/\pageref{LastPage}}
}

%% Redefine the fancy style so that \chapter and \part pages look right.
\fancypagestyle{fancyisg}{
  \fancyhf{}                              % Flush all current headers
  \renewcommand{\headrulewidth}{0.4pt}    % Enable the header bar
  \fancyhead[L]{\small\pplfamily\textsc{\doctype}}
  \fancyhead[C]{\lmssfamily\isg}
  \fancyhead[R]{\small\putfamily\textsc{\nouppercase{\rightmark}}}
  \renewcommand{\footrulewidth}{0.4pt}    % Enable the header bar
  \fancyfoot[L]{\scriptsize\phvfamily\docauthor~(\docstudentnumber). \eece. \up}
  \fancyfoot[R]{\footnotesize\putfamily\thepage/\pageref{LastPage}}
}

%% Define automatic macro that gets applied at all font changes to reset
%%   the spacing before and after equations.
\RequirePackage{etoolbox}
\newcommand{\zerodisplayskips}{%
  \setlength{\abovedisplayskip}{4.5pt plus 1pt minus 1pt}
  \setlength{\belowdisplayskip}{4.5pt plus 1pt minus 1pt}
  \setlength{\abovedisplayshortskip}{0pt plus 1pt}
  \setlength{\belowdisplayshortskip}{0pt minus 1pt}
}
\appto{\normalsize}{\zerodisplayskips}
\appto{\small}{\zerodisplayskips}
\appto{\footnotesize}{\zerodisplayskips}

%% Include space between rows in equarray
\setlength{\jot}{5 pt}

%% Allow extra whitespace in lines, decrease number of hyphenations
\sloppy

% Enable the generation of an Index
\makeindex

% Local macros
\newcommand{\isg}{\mbox{\lmssfamily\ \hspace{-4pt}ISG@UP}\xspace}
\newcommand{\up}{University of Pretoria.}
\newcommand{\eeceup}{University of Pretoria. Department of Electrical, %
    Electronic and Computer Engineering.}
\newcommand{\eece}{Department of Electrical, Electronic and Computer %
    Engineering}

%% Define block styles for tikz.
\tikzstyle{decision} = [diamond, draw, fill=blue!20,
    text width=4.5em, text badly centered, node distance=3cm, inner sep=0pt]
\tikzstyle{block} = [rectangle, draw, fill=blue!20,
    text width=6em, text centered, rounded corners, minimum height=4em]
\tikzstyle{line} = [draw, -latex']
\tikzstyle{cloud} = [draw, ellipse,fill=red!20, node distance=3cm,
    minimum height=2em]

% Pull in support for EPS -> PDF conversion for the UP logo.
\ifpdf{}
  \usepackage{epstopdf}
  \pdfadjustspacing=1
\else
\fi

%% Disable extra line spacing.
\renewcommand{\baselinestretch}{1}

\endinput

%% End of File.
