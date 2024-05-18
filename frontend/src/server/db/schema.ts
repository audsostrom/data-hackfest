import { relations, sql } from "drizzle-orm";
import {
  doublePrecision,
  index,
  integer, pgEnum,
  pgTableCreator,
  primaryKey,
  serial,
  text,
  timestamp,
  varchar,
} from "drizzle-orm/pg-core";
import { type AdapterAccount } from "next-auth/adapters";

export const mpaaRatingEnum = pgEnum("mpaa_rating", ['G', 'PG', 'PG-13', 'R', 'NC-17']);

/**
 * This is an example of how to use the multi-project schema feature of Drizzle ORM. Use the same
 * database instance for multiple projects.
 *
 * @see https://orm.drizzle.team/docs/goodies#multi-project-schema
 */
export const createTable = pgTableCreator((name) => `frontend_${name}`);

export const posts = createTable(
  "post",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 256 }),
    createdById: varchar("createdById", { length: 255 })
      .notNull()
      .references(() => users.id),
    createdAt: timestamp("created_at", { withTimezone: true })
      .default(sql`CURRENT_TIMESTAMP`)
      .notNull(),
    updatedAt: timestamp("updatedAt", { withTimezone: true }),
  },
  (example) => ({
    createdByIdIdx: index("createdById_idx").on(example.createdById),
    nameIndex: index("name_idx").on(example.name),
  })
);

export const users = createTable("user", {
  id: varchar("id", { length: 255 }).notNull().primaryKey(),
  username: varchar("username", { length: 255 }),
  name: varchar("name", { length: 255 }),
  email: varchar("email", { length: 255 }).notNull(),
  emailVerified: timestamp("emailVerified", {
    mode: "date",
    withTimezone: true,
  }).default(sql`CURRENT_TIMESTAMP`),
  image: varchar("image", { length: 255 }),
});

export const usersRelations = relations(users, ({ many }) => ({
  accounts: many(accounts),
  reviews: many(reviews),
}));

export const accounts = createTable(
  "account",
  {
    userId: varchar("userId", { length: 255 })
      .notNull()
      .references(() => users.id),
    type: varchar("type", { length: 255 })
      .$type<AdapterAccount["type"]>()
      .notNull(),
    provider: varchar("provider", { length: 255 }).notNull(),
    providerAccountId: varchar("providerAccountId", { length: 255 }).notNull(),
    refresh_token: text("refresh_token"),
    access_token: text("access_token"),
    expires_at: integer("expires_at"),
    token_type: varchar("token_type", { length: 255 }),
    scope: varchar("scope", { length: 255 }),
    id_token: text("id_token"),
    session_state: varchar("session_state", { length: 255 }),
  },
  (account) => ({
    compoundKey: primaryKey({
      columns: [account.provider, account.providerAccountId],
    }),
    userIdIdx: index("account_userId_idx").on(account.userId),
  })
);

export const accountsRelations = relations(accounts, ({ one }) => ({
  user: one(users, { fields: [accounts.userId], references: [users.id] }),
}));

export const sessions = createTable(
  "session",
  {
    sessionToken: varchar("sessionToken", { length: 255 })
      .notNull()
      .primaryKey(),
    userId: varchar("userId", { length: 255 })
      .notNull()
      .references(() => users.id),
    expires: timestamp("expires", {
      mode: "date",
      withTimezone: true,
    }).notNull(),
  },
  (session) => ({
    userIdIdx: index("session_userId_idx").on(session.userId),
  })
);

export const sessionsRelations = relations(sessions, ({ one }) => ({
  user: one(users, { fields: [sessions.userId], references: [users.id] }),
}));

export const verificationTokens = createTable(
  "verificationToken",
  {
    identifier: varchar("identifier", { length: 255 }).notNull(),
    token: varchar("token", { length: 255 }).notNull(),
    expires: timestamp("expires", {
      mode: "date",
      withTimezone: true,
    }).notNull(),
  },
  (vt) => ({
    compoundKey: primaryKey({ columns: [vt.identifier, vt.token] }),
  })
);

export const productions = createTable(
  "production",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 255 }).notNull(),
  }
);
export const productionRelations = relations(productions, ({ many }) => ({
  movieProductions: many(movieProductions),
}));


export const movies = createTable(
  "movie",
  {
    id: serial("id").primaryKey(),
    title: varchar("title", { length: 255 }).notNull(),
    genres: text("genres"),
    length: integer("length").notNull(),
    release_dt: timestamp("release_dt", { withTimezone: true }).notNull(),
    synopsis: text("synopsis"),
    vote_avg: integer("vote_avg"),
    vote_count: integer("vote_count"),
    lang: varchar("lang", { length: 255 }),
    mpaa_rating: mpaaRatingEnum("mpaa_rating"),
  },
);

export type Movie = typeof movies.$inferSelect;
export type NewMovie = typeof movies.$inferInsert;

export const movieRelations = relations(movies, ({ many }) => ({
  movieProductions: many(movieProductions),
  reviews: many(reviews),
}));

export const movieProductions = createTable(
  "movie_production",
  {
    movieId: integer("movieId").notNull().references(() => movies.id),
    productionId: integer("productionId").notNull().references(() => productions.id),
  },
);

export const movieProductionsRelations = relations(movieProductions, ({ one }) => ( {
  movie: one(movies, {
    fields: [movieProductions.movieId],
    references: [movies.id],
  }),
  production: one(productions, {
    fields: [movieProductions.productionId],
    references: [productions.id],
  }),
}));

export const reviews = createTable(
  "reviews",
  {
    id: serial("id").primaryKey(),
    userId: varchar("userId", { length: 255 })
    .notNull()
    .references(() => users.id),
    movieId: integer("movieId").notNull().references(() => movies.id), 
    desc: text("desc"),
    rating: doublePrecision("rating").notNull(),
  },
);

export const reviewsRelations = relations(reviews, ({ one }) => ( {
  movie: one(movies, {
    fields: [reviews.movieId],
    references: [movies.id],
  }),
  user: one(users, {
    fields: [reviews.userId],
    references: [users.id],
  }),
}));

export const favorites = createTable(
  "favorite",
  {
    userId: varchar("userId", { length: 255 })
      .notNull()
      .references(() => users.id),
    movieId: integer("movieId").notNull().references(() => movies.id),
  },
);

export type Favorite = typeof favorites.$inferSelect;
export type NewFavorite = typeof favorites.$inferInsert;

export const favoritesRelations = relations(favorites, ({ one }) => ( {
  movie: one(movies, {
    fields: [favorites.movieId],
    references: [movies.id],
  }),
  user: one(users, {
    fields: [favorites.userId],
    references: [users.id],
  }),
}));

export const languages = createTable(
  "language",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 255 }).notNull(),
  },
);

export type Language = typeof languages.$inferSelect;
export type NewLanguage = typeof languages.$inferInsert;

export const movieLanguages = createTable(
  "movie_language",
  {
    movieId: integer("movieId").notNull().references(() => movies.id),
    languageId: integer("languageId").notNull().references(() => languages.id),
  },
);

export const genres = createTable(
  "genre",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 255 }).notNull(),
  },
);

export type Genre = typeof genres.$inferSelect;
export type NewGenre = typeof genres.$inferInsert;

export const movieGenres = createTable(
  "movie_genre",
  {
    movieId: integer("movieId").notNull().references(() => movies.id),
    genreId: integer("genreId").notNull().references(() => genres.id),
  },
);

export const keywords = createTable(
  "keyword",
  {
    id: serial("id").primaryKey(),
    name: varchar("name", { length: 255 }).notNull(),
  },
);

export type Keyword = typeof keywords.$inferSelect;
export type NewKeyword = typeof keywords.$inferInsert;

export const movieKeywords = createTable(
  "movie_keyword",
  {
    movieId: integer("movieId").notNull().references(() => movies.id),
    keywordId: integer("keywordId").notNull().references(() => keywords.id),
  },
);

