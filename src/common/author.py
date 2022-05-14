class Author(object):
    count = 0

    def __init__(self, name, id,fos,team_authors):
        Author.count += 1
        self.id = id
        self.name = name
        self.fos = fos
        self.team_authors = team_authors
        self.teams = self.set_teams()
        self.skills = self.set_skills()

    def get_author_id(self):
        return self.id
    def get_author_popularity(self):
        return self.count
    def set_skills(self):
        skills = set()
        for skill in self.fos:
            skills.add(skill["name"].replace(" ", "_"))
        return skills

    def get_skills(self):
        return self.skills

    def set_teams(self):
        teams = set()
        if len(self.team_authors) > 1:
            for author in self.team_authors:
                if(self.name!= author["name"]):
                    teams.add(author["name"])
        return teams
    def get_teams(self):
        return self.teams
