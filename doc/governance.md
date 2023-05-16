
# Abstract

The purpose of this document is to formalize the governance process used by the
`HyperSpy` project, to clarify how decisions are made and how the various
elements of our community interact. This document establishes a decision-making
structure that takes into account feedback from all members of the community
and strives to find consensus, while avoiding any deadlocks.

This is a meritocratic, consensus-based community project. Anyone with an interest
in the project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place and how to set about earning merit within the project community.

The foundations of Project governance are:

- Openness & Transparency
- Active Contribution
- Institutional Neutrality

# Historical Considerations

Project leadership was initially provided by its main creator, Francisco de la Peña.
Additional leadership has also been provided by a subset of Contributors, called
core developers, whose significant or consistent contributions have been recognized
by their receiving “commit rights” to the Project repositories. While this approach
has served us well, as the Project grows, we see a need for a more formal
governance model. Moving forward, the Project leadership will consist of a
Steering Council and a Project Leader. We view this governance model as the
formalization of what we are already doing, rather than a change in direction.

# Roles And Responsibilities

## The Community

The HyperSpy community consists of anyone using or working with the project
in any way.

## Contributors

A community member can become a contributor by interacting directly with the
project in concrete ways, such as:

- proposing a change to the code via a
  [GitHub pull request](https://github.com/hyperspy/hyperspy/pulls);
- reporting issues on our
  [GitHub issues page](https://github.com/hyperspy/hyperspy/issues);
- proposing a change to the documentation, or
  [demos](https://github.com/hyperspy/hyperspy/hyperspy-demos) via a
  GitHub pull request;
- discussing the design of HyperSpy in existing
  [issues](https://github.com/hyperspy/hyperspy/issues) and
  [pull requests](https://github.com/hyperspy/hyperspy/pulls);
- reviewing [open pull requests](https://github.com/hyperspy/hyperspy/pulls)

among other possibilities. Any community member can become a contributor, and
all are encouraged to do so. By contributing to the project, community members
can directly help to shape its future.

Contributors are encouraged to read the [contributing guide](https://github.com/hyperspy/hyperspy/blob/RELEASE_next_minor/.github/CONTRIBUTING.md).

## Core developers

Core developers are trusted community members that have demonstrated continued
commitment to the project through ongoing contributions. They
have shown they can be trusted to maintain HyperSpy with care. Becoming a
core developer allows contributors to merge approved pull requests, and thereby more easily carry on with their project related
activities. Core developers appear as organization members on the HyperSpy
[GitHub organization](https://github.com/orgs/hyperspy/people) and are on our
[@hyperspy/developers](https://github.com/orgs/hyperspy/teams/developers) GitHub
team. Core developers are expected to review code contributions while adhering to the
[core developer guide](CORE_DEV_GUIDE.md). New core developers can be nominated
by any existing core developer, and for details on that process see our core
developer guide.

Core developers that have not contributed to the project (commits or GitHub comments)
in the past 12 months will be asked if they want to become emeritus core developers
and recant their commit until they become active again.
The list of core developers, active and emeritus is public.

## BDFL

The Project’s Benevolent dictator for life (BDFL) is Francisco De La Peña.
The BDFL has the authority to make all final decisions for The Project. In practice
the BDFL chooses to defer that authority to the consensus of the Steering Council
(see below). It is expected that the BDFL will only rarely assert his final authority.
Because rarely used, we refer to BDFL’s final authority as a “special” or “overriding” vote. When it does occur, the BDFL override typically happens in situations where there is a deadlock in the Steering Council (see below) or if the Steering Council asks the BDFL to make a decision on a specific matter.

## Steering Council

The Project will have a Steering Council that consists of Project Contributors
who have produced contributions that are substantial in quality and quantity,
and sustained over at least one year. The overall role of the Council is to ensure,
with input from the Community the long-term well-being of the project, both technically and as a community.

During the everyday project activities, council members participate in all
discussions, code review and other project activities as peers with all other
Contributors and the Community. In these everyday activities, Council Members do
not have any special power or privilege through their membership on the Council.
However, it is expected that because of the quality and quantity of their contributions
and their expert knowledge of the Project Software and Services that Council Members
will provide useful guidance, both technical and in terms of project direction,
to potentially less experienced contributors.

The Steering Council and its Members play a special role in certain situations.
In particular, the Council may:

- Make decisions about the overall scope, vision and direction of the project.
- Make decisions about strategic collaborations with other organizations or individuals.
- Make decisions about specific technical issues, features, bugs and pull requests.
  They are the primary mechanism of guiding the code review process and merging pull requests.
- Make decisions when regular community discussion doesn’t produce consensus on
  an issue in a reasonable time frame.

The chair of the Steering Council is elected by the Steering Council and voted every year. The chair may delegate their authority on a particular decision or set of decisions to any other Council member at their discretion. The chair is responsible for ensuring that all Steering Council activities that require a vote are properly documented.

### Council membership

The Steering Council is currently fixed in size [to be defined] members. This number may increase in the future. The initial Steering Council (in alphabetical order) consists of [to be determined].
To become eligible for being a Steering Council Member an individual must be a
Project Contributor who has produced contributions that are substantial in quality
and quantity, and sustained over at least one year. Potential Council Members are
nominated by existing Council members and voted upon by the existing Council.

When considering potential Members, the Council will look at candidates with a
comprehensive view of their contributions. This will include but is not limited
to code, code review, infrastructure work, chat participation, community
help/building, education and outreach, design work, etc. We are deliberately
not setting arbitrary quantitative metrics (like “100 commits in this repo”) to
avoid encouraging behavior that plays to the metrics rather than the project’s
overall well-being. We want to encourage a diverse array of backgrounds, viewpoints
and talents in our team, which is why we explicitly do not define code as the
sole metric on which council membership will be evaluated.

If a Council member becomes inactive in the project for a period of one year,
they will be considered for removal from the Council. Before removal, inactive
Member will be approached by the BDFL to see if they plan
on returning to active participation. If not, they will be removed immediately
upon a Council vote. If they plan on returning to active participation soon,
they will be given a grace period of one year. If they don’t return to active
participation within that time period they will be removed by vote of the Council
without further grace period. All former Council members can be considered for
membership again at any time in the future, like any other Project Contributor.
Retired Council members will be listed on the project website, acknowledging the
period during which they were active in the Council.

The Council reserves the right to eject current Members, other than the BDFL,
if they are deemed to be actively harmful to the project’s well-being, and
attempts at communication and conflict resolution have failed.

The HyperSpy steering council may be contacted at `EMAIL ADDRESS` or via the
[@hyperspy/steering-council](https://github.com/orgs/hyperspy/teams/steering-council) GitHub team.

## Conflict of interest

It is expected that the SC Members will be employed at a wide range of companies,
universities and non-profit organizations. Because of this, it is possible that
Members will have conflict of interests. Such conflict of interests include,
but are not limited to:

- Financial interests, such as investments, employment, or contracting work,
  outside of The Project that may influence their work on The Project.
- Access to proprietary information of their employer that could potentially
  leak into their work with the Project.
- An issue where the person privately gains an advantage from The Project resources,
  but The Project has no gain or suffers a disadvantage.
  
All members of the Steering Council, shall disclose to the rest of the Council
any conflict of interest they may have. Members with a conflict of interest in
a particular issue may participate in Council discussions on that issue,
but must recuse themselves from voting on the issue. If the Project Lead has recused
themself for a particular decision, they will appoint a substitute BDFL for that decision.

# Decision Making Process

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the [issue tracker](https://github.com/hyperspy/hyperspy/issues). Occasionally,
sensitive discussion may occur on a private communication channel.

Decisions should be made in accordance with the [mission and values](MISSION_AND_VALUES.md)
of the HyperSpy project.

The HyperSpy uses a “consensus seeking” process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
Core developers are expected to distinguish between fundamental objections to a
proposal and minor perceived flaws that they can live with, and not hold up the
decision-making process for the latter.  If no option can be found without
objections, the decision is escalated to the SC, which will itself use
consensus seeking to come to a resolution. In the unlikely event that there is
still a deadlock, the proposal will move forward if it has the support of a
simple majority of the SC.

Decisions (in addition to adding core developers and SC membership as above)
are made according to the following rules:

- **Minor documentation changes**, such as typo fixes, or addition / correction of a
  sentence, require approval by a core developer *and* no disagreement or requested
  changes by a core developer on the issue or pull request page (lazy
  consensus). Core developers are expected to give “reasonable time” to others
  to give their opinion on the pull request if they’re not confident others
  would agree.

- **Code changes and major documentation changes** require agreement by *one*
  core developer *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus). For all changes of this type,
  core developers are expected to give “reasonable time” after approval and before
  merging for others to weigh in on the pull request in its final state.

- **Changes to the API principles** require a dedicated issue on our
  [issue tracker](https://github.com/hyperspy/hyperspy/issues) and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a dedicated issue on our [issue tracker](https://github.com/hyperspy/hyperspy/issues)
  and follow the decision-making process outlined above,
  *unless* there is unanimous agreement from core developers on the change in
  which case it can move forward faster.

If an objection is raised on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected by
escalating to the SC.

# Acknowledgements

This document is adapted from the [Jupyter Project governance](https://jupyter.org/governance/governance.html).


