%% MastersDoctoralThesis.cls
%% Compatible with the Universidad del Norte thesis format
%% Based on the standard MastersDoctoralThesis template

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{MastersDoctoralThesis}[2024/01/01 v2.0 Masters/Doctoral Thesis Class]

\newcommand*{\superref}[1]{\textsuperscript{\ref{#1}}}

%% Options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper]{book}

%% Required packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathpazo} % Palatino font (matches example)
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\RequirePackage{xcolor}
\definecolor{darkred}{RGB}{128,0,0}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage[round]{natbib}
\RequirePackage{emptypage}

%% Page geometry
\geometry{
  paper=a4paper,
  inner=3cm,
  outer=2.5cm,
  top=2.5cm,
  bottom=2.5cm,
  headheight=14pt
}

%% Line spacing
\onehalfspacing

%% Header/footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\thepage\hfill\leftmark}
\fancyhead[RO]{\rightmark\hfill\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyhead[R]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

%% Chapter title formatting
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries}
  {\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-20pt}{40pt}

%% Thesis information commands
\newcommand*{\thesistitle}[1]{\def\ttitle{#1}}
\newcommand*{\supervisor}[1]{\def\supname{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\author@name}{}
\renewcommand*{\author}[1]{\def\authorname{#1}\def\author@name{#1}}
\newcommand*{\authordisplay}[1]{\def\authordisplayname{#1}}
\authordisplay{Author Name}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\thesisdate}[1]{\def\tdate{#1}}

%% Default values
\thesistitle{Thesis Title}
\supervisor{Supervisor Name}
\degree{Master of Science}
\author{Author Name}
\university{Universidad del Norte}
\department{Department of Software Engineer\\and\\Computer Science}
\faculty{Engineering Division}
\thesisdate{\today}

%% Title page
\renewcommand{\maketitle}{%
  \begin{titlepage}
  \begin{center}
  \vspace*{1cm}
  {\color{darkred}\scshape\LARGE \href{https://www.uninorte.edu.co/}{\univname} \par}
  \vspace{1cm}
  {\large \textsc{Master Thesis}\par}
  \vspace{0.5cm}
  \rule{\linewidth}{0.5pt}\par
  \vspace{0.5cm}
  {\huge\bfseries \ttitle \par}
  \vspace{0.5cm}
  \rule{\linewidth}{0.5pt}\par
  \vspace{2cm}
  \begin{minipage}[t]{0.4\textwidth}
  \begin{flushleft}
  \emph{Author:}\\
  {\color{darkred}\href{https://andremov.dev}{\authordisplayname}}
  \end{flushleft}
  \end{minipage}
  \begin{minipage}[t]{0.4\textwidth}
  \begin{flushright}
  \emph{Supervisor:}\\
  {\color{darkred}\href{https://github.com/enino84}{\supname}}
  \end{flushright}
  \end{minipage}
  \vfill
  {\itshape A thesis submitted in fulfillment of the requirements\\
  for the degree of \degreename}\\[0.3cm]
  {\itshape in the}\\[0.3cm]
  {\color{darkred}\href{https://www.uninorte.edu.co/web/ingenierias}{\deptname}}\\[2cm]
  {\large \tdate}
  \end{center}
  \end{titlepage}
  \cleardoublepage
}

%% Declaration environment
\newenvironment{declaration}{
  \cleardoublepage
  \thispagestyle{plain}
  \vspace*{2cm}
  {\huge\bfseries Declaration of Authorship\par}
  \addcontentsline{toc}{chapter}{Declaration of Authorship}
  \vspace{1.5cm}
}{
  \vspace{2cm}
  \noindent Signed:\\[1.5cm]
  \rule[0.5em]{25em}{0.5pt}\\[0.5cm]
  \noindent Date:\\[1cm]
  \rule[0.5em]{25em}{0.5pt}
  \cleardoublepage
}

%% Abstract environment
\newenvironment{thesis-abstract}{
  \cleardoublepage
  \thispagestyle{plain}
  \begin{center}
  {\color{darkred}\scshape \href{https://www.uninorte.edu.co/}{\univname}}\\[0.5cm]
  {\Large\itshape Abstract}\\[0.5cm]
  \addcontentsline{toc}{chapter}{Abstract}
  {\color{darkred}\href{https://www.uninorte.edu.co/web/ingenierias}{\facname}\\
  \href{https://www.uninorte.edu.co/web/ingenierias}{\deptname}}\\[0.5cm]
  {\degreename}\\[0.5cm]
  {\bfseries \ttitle}\\[0.5cm]
  by \authorname
  \end{center}
  \vspace{1cm}
}{
  \cleardoublepage
}

%% Acknowledgements environment
\newenvironment{acknowledgements}{
  \cleardoublepage
  \thispagestyle{plain}
  \vspace*{2cm}
  {\huge\itshape Acknowledgements\par}
  \addcontentsline{toc}{chapter}{Acknowledgements}
  \vspace{1.5cm}
}{
  \cleardoublepage
}

%% Dedication command
\newcommand{\dedicatory}[1]{
  \cleardoublepage
  \thispagestyle{empty}
  \vspace*{5cm}
  \begin{flushleft}
  \itshape #1
  \end{flushleft}
  \cleardoublepage
}

%% hyperref (loaded last to avoid \author conflict)
\RequirePackage[colorlinks=true,linkcolor=darkred,citecolor=magenta,urlcolor=darkred]{hyperref}
